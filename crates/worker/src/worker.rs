//! Canonical worker types and identity.
//!
//! This module defines the single source of truth for worker identity, type
//! enums, and core configuration. These types are shared across API
//! request/response boundaries and internal runtime state.

use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{Arc, LazyLock},
    time::Duration,
};

#[cfg(feature = "axum")]
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use bitflags::bitflags;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "axum")]
use serde_json::json;

use super::{identity::endpoint::Endpoint, model_card::ModelCard, value_bounds};

// ── Default value constants ──────────────────────────────────────────

pub const DEFAULT_WORKER_PRIORITY: u32 = 50;
pub const DEFAULT_WORKER_COST: f32 = 1.0;

// ── Enums ────────────────────────────────────────────────────────────

/// Worker type classification.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum WorkerType {
    /// Regular worker for standard routing.
    #[default]
    Regular,
    /// Prefill worker for PD disaggregated mode.
    Prefill,
    /// Decode worker for PD disaggregated mode.
    Decode,
    /// Encode worker for EPD disaggregated mode: runs the vision tower and
    /// ships image embeddings to a prefill worker over Mooncake.
    Encode,
}

impl WorkerType {
    /// Static string form, identical to `Display`. For hot-path metric labels
    /// that must avoid per-call allocation/interning.
    pub const fn as_str(self) -> &'static str {
        match self {
            WorkerType::Regular => "regular",
            WorkerType::Prefill => "prefill",
            WorkerType::Decode => "decode",
            WorkerType::Encode => "encode",
        }
    }
}

impl std::fmt::Display for WorkerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for WorkerType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("regular") {
            Ok(WorkerType::Regular)
        } else if s.eq_ignore_ascii_case("prefill") {
            Ok(WorkerType::Prefill)
        } else if s.eq_ignore_ascii_case("decode") {
            Ok(WorkerType::Decode)
        } else if s.eq_ignore_ascii_case("encode") {
            Ok(WorkerType::Encode)
        } else {
            Err(format!("Unknown worker type: {s}"))
        }
    }
}

/// Connection mode for worker communication.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ConnectionMode {
    /// HTTP/REST connection.
    #[default]
    Http,
    /// gRPC connection.
    Grpc,
    /// Direct ZMQ connection to an engine core (bypasses the gRPC Python
    /// servicer). Speaks the engine's native ZMQ wire: over `ipc://` sockets
    /// to a same-host engine, over `tcp://` to engines on other hosts.
    Zmq,
}

impl ConnectionMode {
    /// Classify a worker URL by its scheme — the single source of truth for
    /// scheme → connection mode. Returns `None` for a bare `host:port` (no
    /// scheme) or an unrecognized scheme, leaving the default to the caller
    /// (some probe both protocols; some default to HTTP).
    pub fn from_url(url: &str) -> Option<Self> {
        if url.starts_with("ipc://") || url.starts_with("tcp://") {
            Some(ConnectionMode::Zmq)
        } else if url.starts_with("grpc://") || url.starts_with("grpcs://") {
            Some(ConnectionMode::Grpc)
        } else if url.starts_with("http://") || url.starts_with("https://") {
            Some(ConnectionMode::Http)
        } else {
            None
        }
    }

    /// Whether the gateway speaks the engine's wire protocol directly for this
    /// mode. gRPC and direct-ZMQ are distinct transports that both ride the
    /// gRPC-router pipeline (differing only in the client behind it); HTTP is
    /// proxied and has its own router. Callers use this to treat gRPC and ZMQ
    /// workers uniformly.
    pub fn uses_grpc_pipeline(self) -> bool {
        matches!(self, ConnectionMode::Grpc | ConnectionMode::Zmq)
    }

    /// Static string form, identical to `Display`. For hot-path metric labels
    /// that must avoid per-call allocation/interning.
    pub const fn as_str(self) -> &'static str {
        match self {
            ConnectionMode::Http => "http",
            ConnectionMode::Grpc => "grpc",
            ConnectionMode::Zmq => "zmq",
        }
    }
}

impl std::fmt::Display for ConnectionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Worker lifecycle status, modeled after Kubernetes pod conditions.
///
/// Separates "starting up" from "can serve traffic" from "broken" to prevent
/// premature removal of workers that haven't been locally verified.
///
/// Uses `#[repr(u8)]` with explicit discriminants so atomic storage
/// (`AtomicU8`) and wire compatibility do not depend on enum ordering.
#[repr(u8)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStatus {
    /// Just registered, not yet proven healthy locally.
    /// Not routable. Health checker probes until first success.
    #[default]
    Pending = 0,

    /// Locally verified and passing health checks. Routable.
    Ready = 1,

    /// Was Ready, now failing readiness checks. Not routable.
    /// NOT removed — may recover. Health checker continues probing.
    NotReady = 2,

    /// Sustained liveness failure. Will be removed if `--remove-unhealthy-workers`.
    Failed = 3,

    /// Marked for removal (e.g. K8s pod has `deletionTimestamp`). Excluded
    /// from selection so no new traffic is routed, but kept in the registry
    /// so in-flight requests can complete before the worker is removed.
    Draining = 4,
}

impl WorkerStatus {
    /// Try to convert a `u8` discriminant to a `WorkerStatus`.
    /// Returns `None` for unrecognized values.
    pub fn try_from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Pending),
            1 => Some(Self::Ready),
            2 => Some(Self::NotReady),
            3 => Some(Self::Failed),
            4 => Some(Self::Draining),
            _ => None,
        }
    }

    /// Convert a `u8` discriminant to a `WorkerStatus`, falling back to
    /// `Pending` for unrecognized values (with a debug assertion).
    pub fn from_u8(value: u8) -> Self {
        let maybe = Self::try_from_u8(value);
        debug_assert!(
            maybe.is_some(),
            "invalid WorkerStatus discriminant: {value}"
        );
        maybe.unwrap_or(Self::Pending)
    }

    /// Returns `true` if the worker is routable (only `Ready`).
    pub fn is_routable(self) -> bool {
        matches!(self, Self::Ready)
    }
}

impl std::fmt::Display for WorkerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerStatus::Pending => write!(f, "pending"),
            WorkerStatus::Ready => write!(f, "ready"),
            WorkerStatus::NotReady => write!(f, "not_ready"),
            WorkerStatus::Failed => write!(f, "failed"),
            WorkerStatus::Draining => write!(f, "draining"),
        }
    }
}

/// Composite key identifying a group of workers with the same characteristics.
///
/// Groups workers by `(model_id, worker_type, connection_mode)` — the natural
/// partitioning used for metrics, load monitoring, and policy management.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkerGroupKey {
    pub model_id: String,
    pub worker_type: WorkerType,
    pub connection_mode: ConnectionMode,
}

impl std::fmt::Display for WorkerGroupKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.model_id, self.worker_type, self.connection_mode
        )
    }
}

/// Runtime implementation type for workers.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeType {
    /// No runtime type specified — the backend will be auto-detected.
    #[default]
    Unspecified,
    /// SGLang runtime.
    Sglang,
    /// vLLM runtime.
    Vllm,
    /// TensorRT-LLM runtime.
    Trtllm,
    /// MLX runtime (Apple Silicon).
    Mlx,
    /// TokenSpeed runtime.
    TokenSpeed,
    /// Generic OpenAI-compatible HTTP backend whose engine could not be
    /// identified (e.g. a nested SMG gateway fronting the real engine).
    /// Routed as plain OpenAI HTTP; no engine-specific features are assumed.
    Generic,
    /// External OpenAI-compatible API (not local inference).
    External,
}

impl RuntimeType {
    /// Returns `true` when the caller supplied an explicit runtime type.
    pub fn is_specified(self) -> bool {
        !matches!(self, RuntimeType::Unspecified)
    }

    /// Static string form, identical to `Display`. For hot-path metric labels
    /// that must avoid per-call allocation/interning.
    pub fn as_str(self) -> &'static str {
        match self {
            RuntimeType::Unspecified => "unspecified",
            RuntimeType::Sglang => "sglang",
            RuntimeType::Vllm => "vllm",
            RuntimeType::Trtllm => "trtllm",
            RuntimeType::Mlx => "mlx",
            RuntimeType::TokenSpeed => "tokenspeed",
            RuntimeType::Generic => "generic",
            RuntimeType::External => "external",
        }
    }
}

impl std::fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for RuntimeType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("unspecified") {
            Ok(RuntimeType::Unspecified)
        } else if s.eq_ignore_ascii_case("sglang") {
            Ok(RuntimeType::Sglang)
        } else if s.eq_ignore_ascii_case("vllm") {
            Ok(RuntimeType::Vllm)
        } else if s.eq_ignore_ascii_case("trtllm") || s.eq_ignore_ascii_case("tensorrt-llm") {
            Ok(RuntimeType::Trtllm)
        } else if s.eq_ignore_ascii_case("mlx") {
            Ok(RuntimeType::Mlx)
        } else if s.eq_ignore_ascii_case("tokenspeed") {
            Ok(RuntimeType::TokenSpeed)
        } else if s.eq_ignore_ascii_case("generic") {
            Ok(RuntimeType::Generic)
        } else if s.eq_ignore_ascii_case("external") {
            Ok(RuntimeType::External)
        } else {
            Err(format!("Unknown runtime type: {s}"))
        }
    }
}

/// Provider type for external API transformations.
///
/// Different providers have different API formats and requirements.
/// `None` (when used as `Option<ProviderType>`) means native/passthrough —
/// no transformation needed (local engine backends).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI API — strip the engine-specific fields.
    #[serde(alias = "openai")]
    OpenAI,
    /// xAI/Grok — special handling for input items.
    #[serde(alias = "xai", alias = "grok")]
    #[expect(
        clippy::upper_case_acronyms,
        reason = "xAI is a proper company name; XAI matches industry convention and existing serde aliases"
    )]
    XAI,
    /// Anthropic Claude — different API format.
    #[serde(alias = "anthropic", alias = "claude")]
    Anthropic,
    /// Google Gemini — special logprobs handling.
    #[serde(alias = "gemini", alias = "google")]
    Gemini,
    /// Custom provider with string identifier.
    #[serde(untagged)]
    Custom(String),
}

impl ProviderType {
    /// Get provider name as string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenAI => "openai",
            Self::XAI => "xai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Detect provider from URL host.
    /// Returns `None` for URLs that don't match known providers or can't be parsed.
    pub fn from_url(url: &str) -> Option<Self> {
        let host = url::Url::parse(url).ok()?.host_str()?.to_lowercase();

        if host.ends_with("openai.com") {
            Some(Self::OpenAI)
        } else if host.ends_with("x.ai") {
            Some(Self::XAI)
        } else if host.ends_with("anthropic.com") {
            Some(Self::Anthropic)
        } else if host.ends_with("googleapis.com") {
            Some(Self::Gemini)
        } else {
            None
        }
    }

    /// Environment variable name for per-provider admin API key (model discovery).
    /// Returns `None` for `Custom` providers since there's no known env var.
    pub fn admin_key_env_var(&self) -> Option<&'static str> {
        match self {
            Self::OpenAI => Some("OPENAI_ADMIN_KEY"),
            Self::XAI => Some("XAI_ADMIN_KEY"),
            Self::Anthropic => Some("ANTHROPIC_ADMIN_KEY"),
            Self::Gemini => Some("GEMINI_ADMIN_KEY"),
            Self::Custom(_) => None,
        }
    }

    /// Whether this provider uses `x-api-key` header instead of `Authorization: Bearer`.
    pub fn uses_x_api_key(&self) -> bool {
        matches!(self, Self::Anthropic)
    }

    /// Detect provider from model name (heuristic fallback).
    /// Returns `None` for models that don't match known external providers.
    pub fn from_model_name(model: &str) -> Option<Self> {
        let model_lower = model.to_lowercase();
        if model_lower.starts_with("grok") {
            Some(Self::XAI)
        } else if model_lower.starts_with("gemini") {
            Some(Self::Gemini)
        } else if model_lower.starts_with("claude") {
            Some(Self::Anthropic)
        } else if model_lower.starts_with("gpt")
            || model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
        {
            Some(Self::OpenAI)
        } else {
            None
        }
    }
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Serde default helpers ────────────────────────────────────────────

fn default_priority() -> u32 {
    DEFAULT_WORKER_PRIORITY
}

fn default_cost() -> f32 {
    DEFAULT_WORKER_COST
}

fn default_health_check_timeout() -> u64 {
    30
}

fn default_health_check_interval() -> u64 {
    60
}

fn default_health_success_threshold() -> u32 {
    2
}

fn default_health_failure_threshold() -> u32 {
    3
}

fn default_max_connection_attempts() -> u32 {
    20
}

fn default_drain_settle_secs() -> u64 {
    5
}

// ── Health check config ─────────────────────────────────────────────

/// Health check configuration shared across protocol and runtime layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HealthCheckConfig {
    /// Health check timeout in seconds (default: 30).
    #[serde(default = "default_health_check_timeout")]
    pub timeout_secs: u64,

    /// Health check interval in seconds (default: 60).
    #[serde(default = "default_health_check_interval")]
    pub check_interval_secs: u64,

    /// Number of successful health checks needed to mark worker as healthy (default: 2).
    #[serde(default = "default_health_success_threshold")]
    pub success_threshold: u32,

    /// Number of failed health checks before marking worker as unhealthy (default: 3).
    #[serde(default = "default_health_failure_threshold")]
    pub failure_threshold: u32,

    /// Disable periodic health checks for this worker (default: false).
    #[serde(default)]
    pub disable_health_check: bool,

    /// Seconds to keep a worker in `Draining` after `RemoveWorker` is
    /// submitted before the registry entry is actually removed. Lets
    /// in-flight requests complete naturally (default: 5). Set to `0`
    /// to skip draining and remove immediately.
    #[serde(default = "default_drain_settle_secs")]
    pub drain_settle_secs: u64,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            timeout_secs: default_health_check_timeout(),
            check_interval_secs: default_health_check_interval(),
            success_threshold: default_health_success_threshold(),
            failure_threshold: default_health_failure_threshold(),
            disable_health_check: false,
            drain_settle_secs: default_drain_settle_secs(),
        }
    }
}

// ── Worker models ───────────────────────────────────────────────────

/// Models configuration for a worker.
///
/// Encodes the three real cases instead of relying on `Vec` semantics:
/// - `Wildcard` — accepts any model (empty models list on the wire)
/// - `Single` — serves exactly one model
/// - `Multi` — serves multiple distinct models (len >= 2)
#[derive(Debug, Clone, Default)]
pub enum WorkerModels {
    /// Worker accepts any model (e.g., external API without discovery).
    #[default]
    Wildcard,
    /// Worker serves exactly one model (most common for local inference).
    Single(Box<ModelCard>),
    /// Worker serves multiple distinct models (len >= 2).
    Multi(Vec<ModelCard>),
}

impl WorkerModels {
    /// Returns `true` if this is a wildcard (accepts any model).
    pub fn is_wildcard(&self) -> bool {
        matches!(self, Self::Wildcard)
    }

    /// Returns the primary model: `Single` → `Some`, `Multi` → first, `Wildcard` → `None`.
    pub fn primary(&self) -> Option<&ModelCard> {
        match self {
            Self::Wildcard => None,
            Self::Single(card) => Some(card.as_ref()),
            Self::Multi(cards) => cards.first(),
        }
    }

    /// Returns all models as a slice (empty for `Wildcard`).
    pub fn all(&self) -> &[ModelCard] {
        match self {
            Self::Wildcard => &[],
            Self::Single(card) => std::slice::from_ref(card.as_ref()),
            Self::Multi(cards) => cards,
        }
    }

    /// Find a model by ID (checks aliases via `ModelCard::matches`).
    pub fn find(&self, id: &str) -> Option<&ModelCard> {
        match self {
            Self::Wildcard => None,
            Self::Single(card) => card.matches(id).then_some(card.as_ref()),
            Self::Multi(cards) => cards.iter().find(|m| m.matches(id)),
        }
    }

    /// Returns `true` if the worker supports the given model ID.
    /// Wildcard workers always return `true`.
    pub fn supports(&self, id: &str) -> bool {
        match self {
            Self::Wildcard => true,
            _ => self.find(id).is_some(),
        }
    }

    /// Iterate over all models. Empty iterator for `Wildcard`.
    pub fn iter(&self) -> impl Iterator<Item = &ModelCard> {
        self.all().iter()
    }
}

impl From<Vec<ModelCard>> for WorkerModels {
    fn from(models: Vec<ModelCard>) -> Self {
        match models.len() {
            0 => Self::Wildcard,
            1 => {
                let Some(model) = models.into_iter().next() else {
                    return Self::Wildcard;
                };
                Self::Single(Box::new(model))
            }
            _ => Self::Multi(models),
        }
    }
}

/// Serialize as `Vec<ModelCard>` for wire compatibility.
impl Serialize for WorkerModels {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.all().serialize(serializer)
    }
}

/// Deserialize from `Vec<ModelCard>` for wire compatibility.
impl<'de> Deserialize<'de> for WorkerModels {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let models = Vec::<ModelCard>::deserialize(deserializer)?;
        Ok(Self::from(models))
    }
}

/// JsonSchema: wire format is `Vec<ModelCard>`.
impl JsonSchema for WorkerModels {
    fn schema_name() -> Cow<'static, str> {
        "WorkerModels".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<ModelCard>::json_schema(generator)
    }
}

/// When the gateway hands a worker's engine a grammar: only when the request
/// asks for one (a forced tool choice or a JSON response format), which every
/// worker does unless told otherwise, or on every turn, for a model trained
/// to be read through one. Set on the worker spec, as the `constrained_decoding`
/// worker label, or reported by the engine's servicer in server info; the spec
/// wins over the labels, and a configured label over a discovered one
/// ([`ConstrainedDecoding::of`]).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ConstrainedDecoding {
    /// A grammar only when the request asks for one.
    #[default]
    Requested,
    /// A grammar on every turn.
    Always,
}

impl ConstrainedDecoding {
    /// The worker label that carries the policy when it is not on the spec.
    pub const LABEL: &'static str = "constrained_decoding";
    /// The label's value for [`Self::Requested`].
    pub const REQUESTED: &'static str = "requested";
    /// The label's value for [`Self::Always`].
    pub const ALWAYS: &'static str = "always";

    /// The policy a label value names, or why it names none.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            Self::REQUESTED => Ok(Self::Requested),
            Self::ALWAYS => Ok(Self::Always),
            other => Err(format!(
                "constrained_decoding must be '{}' or '{}', not '{other}'",
                Self::REQUESTED,
                Self::ALWAYS
            )),
        }
    }

    /// The policy a worker's `constrained_decoding` label names; anything
    /// else, the label's absence included, is [`Self::Requested`].
    pub fn from_label(value: Option<&str>) -> Self {
        value
            .and_then(|value| Self::parse(value).ok())
            .unwrap_or_default()
    }

    /// The label value for the policy.
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Requested => Self::REQUESTED,
            Self::Always => Self::ALWAYS,
        }
    }

    /// A worker's policy: the spec's when set, else its label's.
    pub fn of(spec: &WorkerSpec) -> Self {
        spec.constrained_decoding
            .unwrap_or_else(|| Self::from_label(spec.labels.get(Self::LABEL).map(String::as_str)))
    }
}

/// The vocabulary of structural-tag nodes a worker's grammar compiler
/// takes, from its `structural_tag_vocabulary` label: the current one unless
/// the label says `older`, for a compiler that predates the `optional`,
/// `plus` and `star` nodes and refuses a tag that carries one. The tables'
/// tags go to such a worker spelled without them, the same language, which
/// every compiler takes; a worker without the label gets the tag as the
/// table built it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TagVocabulary {
    /// Every node the tables use.
    #[default]
    Current,
    /// Without the `optional`, `plus` and `star` nodes; a `token` node stays
    /// as derived.
    Older,
}

impl TagVocabulary {
    /// The worker label that carries it.
    pub const LABEL: &'static str = "structural_tag_vocabulary";
    /// The label's value for [`Self::Older`].
    pub const OLDER: &'static str = "older";

    /// The vocabulary a worker's `structural_tag_vocabulary` label names;
    /// anything else, the label's absence included, is [`Self::Current`].
    pub fn from_label(value: Option<&str>) -> Self {
        match value {
            Some(Self::OLDER) => Self::Older,
            _ => Self::Current,
        }
    }
}

/// Where the engine that runs a forced call's grammar starts applying it on a
/// prompt that ends inside the model's thinking block, as its worker
/// advertises it (the `constrained_decoding_start` label).
///
/// An engine's own server picks the grammar's shape from this: it sends
/// xgrammar's reasoning-aware tags (the thought, its close, then the calls)
/// when it runs no reasoning parser and the bare calls when it does, because
/// its grammar backend then waits for the thought's end; an engine with a
/// reasoning parser set (and `enable_in_reasoning` off) waits the same way.
/// The gateway reads the same fact from the worker and shapes the calls'
/// prefix accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrammarStart {
    /// The worker does not say (an older servicer, an engine without the
    /// label): free text with nothing owed at its end precedes the calls,
    /// which fits an engine of either kind.
    #[default]
    Unknown,
    /// The engine applies the grammar only once its reasoning parser has
    /// seen the model's own `</think>`: nothing precedes the calls, so the
    /// call is forced right at that gate, as the engine's own server does.
    AfterReasoning,
    /// The engine applies the grammar from the first generated token: the
    /// thought, its close and then the calls, so the call is not forced into
    /// the thought.
    FirstToken,
}

impl GrammarStart {
    /// The worker label that carries it.
    pub const LABEL: &'static str = "constrained_decoding_start";
    /// The label's value for [`Self::AfterReasoning`].
    pub const AFTER_REASONING: &'static str = "after_reasoning";
    /// The label's value for [`Self::FirstToken`].
    pub const FIRST_TOKEN: &'static str = "first_token";

    /// The start a worker's `constrained_decoding_start` label names; anything
    /// else, the label's absence included, is [`Self::Unknown`].
    pub fn from_label(value: Option<&str>) -> Self {
        match value {
            Some(Self::AFTER_REASONING) => Self::AfterReasoning,
            Some(Self::FIRST_TOKEN) => Self::FirstToken,
            _ => Self::Unknown,
        }
    }
}

// ── Core identity ────────────────────────────────────────────────────

const REDACTED_API_KEY: &str = "<redacted>";

/// Core worker identity and configuration.
///
/// The single canonical representation of "what is a worker". Used as the
/// shared sub-struct across API requests, API responses, and internal runtime
/// state via `#[serde(flatten)]`.
///
/// Fields use `#[serde(default)]` so the same struct works for both input
/// (partial config from user) and output (fully resolved state).
#[serde_with::skip_serializing_none]
#[derive(Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerSpec {
    /// Worker URL.
    pub url: String,

    /// Models this worker can serve.
    #[serde(default, skip_serializing_if = "WorkerModels::is_wildcard")]
    pub models: WorkerModels,

    /// Worker type: regular, prefill, or decode.
    #[serde(default)]
    pub worker_type: WorkerType,

    /// Connection mode: http or grpc.
    #[serde(default)]
    pub connection_mode: ConnectionMode,

    /// Runtime type: sglang, vllm, trtllm, or external.
    #[serde(default, alias = "runtime")]
    pub runtime_type: RuntimeType,

    /// External provider for API transformations.
    /// `None` means native/passthrough.
    pub provider: Option<ProviderType>,

    /// Additional labels/tags.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub labels: HashMap<String, String>,

    /// Worker priority (higher = preferred).
    #[serde(default = "default_priority")]
    pub priority: u32,

    /// Worker cost factor (baseline = 1.0).
    #[serde(default = "default_cost")]
    pub cost: f32,

    /// Worker API key. Accepted on input, never included in responses.
    #[serde(default, skip_serializing)]
    pub api_key: Option<String>,

    /// Bootstrap port for prefill workers in PD disaggregated mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,

    /// Bootstrap hostname (derived from URL at construction time).
    #[serde(default, skip)]
    pub bootstrap_host: String,

    /// Base URL without DP rank suffix (for DP-aware workers).
    /// When set, `url` contains the rank-suffixed form (`{base}@{rank}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_base_url: Option<String>,

    /// Data-parallel rank (None = not DP-aware).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<usize>,

    /// Total data-parallel group size (None = not DP-aware).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_size: Option<usize>,

    /// KV connector type (e.g. "MooncakeConnector", "NixlConnector").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_connector: Option<String>,

    /// KV role (e.g. "kv_producer", "kv_consumer", "kv_both").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_role: Option<String>,

    /// KV transfer engine id (the engine's `kv_transfer_config.engine_id`; Mooncake PD).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_engine_id: Option<String>,

    /// KV cache block size (tokens per block) for event-driven routing.
    /// When set, overrides the router-level default for this worker's model.
    /// Typically matches the backend engine's page size (e.g. 16 for one of the engines).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_block_size: Option<usize>,
    /// Explicit PD pairing protocol. A prefill and a decode with this set
    /// pair only when the values are equal, and nothing derived about their
    /// transport, engine version or KV layout is compared. The same value
    /// can arrive as a `pairing_protocol` worker label, or from the engine
    /// itself: the gRPC servicers report their `SMG_PAIRING_PROTOCOL`
    /// environment in server info, so a deployment can inject it into the
    /// engine container. Absent everywhere: the router derives the pairing
    /// descriptor from the worker's discovered labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing_protocol: Option<String>,
    /// When the gateway hands this worker's engine a grammar: `always` for a
    /// model read through one on every turn, `requested` (the default) for a
    /// grammar only when the request asks. The same value can arrive as a
    /// `constrained_decoding` worker label or from the engine's servicer in
    /// server info; this field wins ([`ConstrainedDecoding::of`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained_decoding: Option<ConstrainedDecoding>,

    /// Per-worker health check overrides (partial — only `Some` fields override router defaults).
    #[serde(default, skip_serializing_if = "HealthCheckUpdate::is_empty")]
    pub health: HealthCheckUpdate,

    /// Per-worker HTTP connection pool overrides.
    #[serde(default, skip_serializing_if = "HttpPoolConfig::is_empty")]
    pub http_pool: HttpPoolConfig,

    /// Per-worker resilience overrides (retry + circuit breaker).
    #[serde(default, skip_serializing_if = "ResilienceUpdate::is_empty")]
    pub resilience: ResilienceUpdate,

    /// Maximum connection attempts during worker registration (default: 20).
    #[serde(default = "default_max_connection_attempts")]
    pub max_connection_attempts: u32,

    /// Per-worker load monitor interval override (seconds).
    /// When set, workers in the same group use this interval for load polling.
    /// Falls back to the global `load_monitor_interval_secs` from router config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_monitor_interval_secs: Option<u64>,

    /// Per-worker multimodal tensor transport override (`inline` | `shm` | `auto`).
    /// Overrides the router-level `multimodal_tensor_transport` for this worker
    /// (e.g. force `shm` for a co-located worker, `inline` for a remote one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multimodal_tensor_transport: Option<TransportMode>,

    /// Per-worker minimum multimodal tensor size (bytes) before the SHM transport
    /// is used. Overrides the router-level `multimodal_shm_min_bytes`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multimodal_shm_min_bytes: Option<usize>,

    /// Handshake address the frontend BINDS for this worker's engine handshake.
    /// Only meaningful for `connection_mode: zmq`; must be a `tcp://` address.
    /// When unset, the frontend derives a deterministic port from the worker's
    /// `ipc://` path instead — this override exists for engines that dial a
    /// fixed, pre-agreed address rather than the derived one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zmq_handshake_address: Option<String>,

    /// Per-worker absolute overload threshold overrides (partial — only `Some`
    /// fields override gateway defaults). Either field set enables overload
    /// protection for this worker even when the gateway leaves it off.
    #[serde(default, skip_serializing_if = "OverloadUpdate::is_empty")]
    pub overload: OverloadUpdate,
}

impl std::fmt::Debug for WorkerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // List every field so new fields require a formatting decision.
        let Self {
            url,
            models,
            worker_type,
            connection_mode,
            runtime_type,
            provider,
            labels,
            priority,
            cost,
            api_key,
            bootstrap_port,
            bootstrap_host,
            dp_base_url,
            dp_rank,
            dp_size,
            kv_connector,
            kv_role,
            kv_engine_id,
            kv_block_size,
            pairing_protocol,
            constrained_decoding,
            health,
            http_pool,
            resilience,
            max_connection_attempts,
            load_monitor_interval_secs,
            multimodal_tensor_transport,
            multimodal_shm_min_bytes,
            zmq_handshake_address,
            overload,
        } = self;

        // An address prints in its redacted form: its userinfo can carry
        // a credential. A rank is read out of the address alone that is
        // its base, an `@` and digits: in any other, an `@` and digits at
        // the end are a userinfo and a host, and read as a rank they would
        // print the userinfo. In an IPC address they are bytes of its path.
        let url = if self.url_is_ranked() {
            Endpoint::redacted_str(url)
        } else {
            Endpoint::redacted_str_unranked(url)
        };
        f.debug_struct("WorkerSpec")
            .field("url", &url)
            .field("models", models)
            .field("worker_type", worker_type)
            .field("connection_mode", connection_mode)
            .field("runtime_type", runtime_type)
            .field("provider", provider)
            .field("labels", labels)
            .field("priority", priority)
            .field("cost", cost)
            .field("api_key", &api_key.is_some().then_some(REDACTED_API_KEY))
            .field("bootstrap_port", bootstrap_port)
            .field("bootstrap_host", bootstrap_host)
            .field(
                "dp_base_url",
                &dp_base_url.as_deref().map(Endpoint::redacted_str_unranked),
            )
            .field("dp_rank", dp_rank)
            .field("dp_size", dp_size)
            .field("kv_connector", kv_connector)
            .field("kv_role", kv_role)
            .field("kv_engine_id", kv_engine_id)
            .field("kv_block_size", kv_block_size)
            .field("pairing_protocol", pairing_protocol)
            .field("constrained_decoding", constrained_decoding)
            .field("health", health)
            .field("http_pool", http_pool)
            .field("resilience", resilience)
            .field("max_connection_attempts", max_connection_attempts)
            .field("load_monitor_interval_secs", load_monitor_interval_secs)
            .field("multimodal_tensor_transport", multimodal_tensor_transport)
            .field("multimodal_shm_min_bytes", multimodal_shm_min_bytes)
            .field("zmq_handshake_address", zmq_handshake_address)
            .field("overload", overload)
            .finish()
    }
}

impl WorkerSpec {
    /// Create a new `WorkerSpec` with the given URL and sensible defaults.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            models: WorkerModels::Wildcard,
            worker_type: WorkerType::default(),
            connection_mode: ConnectionMode::default(),
            runtime_type: RuntimeType::default(),
            provider: None,
            labels: HashMap::new(),
            priority: DEFAULT_WORKER_PRIORITY,
            cost: DEFAULT_WORKER_COST,
            api_key: None,
            bootstrap_port: None,
            bootstrap_host: String::new(),
            dp_base_url: None,
            dp_rank: None,
            dp_size: None,
            kv_connector: None,
            kv_role: None,
            kv_engine_id: None,
            kv_block_size: None,
            pairing_protocol: None,
            constrained_decoding: None,
            health: HealthCheckUpdate::default(),
            http_pool: HttpPoolConfig::default(),
            resilience: ResilienceUpdate::default(),
            max_connection_attempts: default_max_connection_attempts(),
            load_monitor_interval_secs: None,
            multimodal_tensor_transport: None,
            multimodal_shm_min_bytes: None,
            zmq_handshake_address: None,
            overload: OverloadUpdate::default(),
        }
    }

    /// Whether [`Self::url`] carries a rank: it is the spec's base
    /// ([`Self::dp_base_url`]), an `@` and digits, as a registration writes
    /// a rank's address beside its base. In any other address an `@` and
    /// digits at the end are a userinfo and a host, or bytes of an IPC
    /// path, and no rank is read out of it.
    pub fn url_is_ranked(&self) -> bool {
        self.dp_base_url
            .as_deref()
            .and_then(|base| self.url.strip_prefix(base))
            .and_then(|rest| rest.strip_prefix('@'))
            .is_some_and(|rank| {
                !rank.is_empty() && rank.bytes().all(|digit| digit.is_ascii_digit())
            })
    }

    /// [`Self::url`] as a request's success path shows it: the registered
    /// string byte for byte where it holds no userinfo, query or fragment,
    /// and its redacted form otherwise ([`Endpoint::redacted_url`]). A
    /// rank is read out of the address by the spec's own fields alone
    /// ([`Self::url_is_ranked`]). In any other address nothing is split
    /// off the end: an `@` and digits there are a userinfo before a host,
    /// which is masked, or bytes of an IPC path, which is written whole.
    pub fn redacted_url(&self) -> String {
        if self.url_is_ranked() {
            Endpoint::redacted_url(&self.url)
        } else {
            Endpoint::redacted_url_unranked(&self.url)
        }
    }
}

/// Multimodal tensor transport mode for large payloads.
///
/// - `Inline`: always carry tensor bytes in the gRPC message.
/// - `Shm`: use same-host `/dev/shm` when SMG can write it.
/// - `Auto`: use `/dev/shm` only when the receiving worker is verified to share
///   SMG's `/dev/shm`; otherwise fall back to inline.
/// - `Rdma`: use the NIXL RDMA pixel lane for large tensors (requires the
///   `mm-rdma` build feature + NIXL); falls back to inline when RDMA is
///   unavailable.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum TransportMode {
    #[default]
    Inline,
    Shm,
    Auto,
    Rdma,
}

impl TransportMode {
    /// Parse from a case-insensitive string (`inline` | `shm` | `auto` | `rdma`).
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "inline" => Some(Self::Inline),
            "shm" => Some(Self::Shm),
            "auto" => Some(Self::Auto),
            "rdma" => Some(Self::Rdma),
            _ => None,
        }
    }

    /// Canonical lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Shm => "shm",
            Self::Auto => "auto",
            Self::Rdma => "rdma",
        }
    }
}

impl std::fmt::Display for TransportMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TransportMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
            .ok_or_else(|| format!("invalid transport mode '{s}'; expected inline|shm|auto|rdma"))
    }
}

/// Where media is fetched and preprocessed for one engine's gRPC workers.
///
/// - `Auto`: forward media references when every worker serving the model
///   advertises worker-side processing; otherwise preprocess on the router.
/// - `Router`: always preprocess on the router (kill switch).
/// - `Worker`: forward references; a request carrying per-item hints is
///   preprocessed on the router, which applies them; fail when no capable
///   worker exists.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum MmProcessingMode {
    #[default]
    Auto,
    Router,
    Worker,
}

impl MmProcessingMode {
    /// Parse from a case-insensitive string (`auto` | `router` | `worker`).
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "router" => Some(Self::Router),
            "worker" => Some(Self::Worker),
            _ => None,
        }
    }

    /// Canonical lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Router => "router",
            Self::Worker => "worker",
        }
    }
}

impl std::fmt::Display for MmProcessingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for MmProcessingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| {
            format!("invalid multimodal processing mode '{s}'; expected auto|router|worker")
        })
    }
}

// ── API types ───────────────────────────────────────────────────────

/// Worker information for API responses.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerInfo {
    /// Worker unique identifier.
    pub id: String,

    /// Primary model ID for backwards compatibility.
    /// Computed from `models[0].id` (single/multi) or `null` (wildcard).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,

    /// Worker identity and configuration.
    ///
    /// Stored behind an `Arc` so building a `WorkerInfo` for a response (e.g.
    /// `GET /workers`) shares the spec instead of deep-cloning it per worker.
    #[serde(flatten)]
    pub spec: Arc<WorkerSpec>,

    /// Whether the worker is healthy.
    pub is_healthy: bool,

    /// Worker lifecycle status (Pending, Ready, NotReady, Failed).
    /// Present alongside `is_healthy` for backward compatibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<WorkerStatus>,

    /// Current load on the worker.
    pub load: usize,

    /// Whether the router speaks HTTP/2 to the worker.
    #[serde(default)]
    pub http2: bool,

    /// The effective PD pairing key of a prefill or decode worker: the
    /// explicit `pairing_protocol`, else the descriptor derived from the
    /// engine's labels (`runtime/transport/layout`; the engine version is
    /// left out because it only counts under strict mode). Prefill and
    /// decode workers pair only within one key. Absent on regular workers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pd_pairing: Option<String>,

    /// Why the gateway's liveness tracker currently keeps the worker out of
    /// routing (`unreachable` or `wedged`) while its health status stands;
    /// absent when it is routable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stalled: Option<String>,

    /// The worker's last polled engine load, as published by the load
    /// monitor. `None` when load monitoring has produced nothing for this
    /// worker yet. Unrelated to `load` above, which counts in-flight
    /// requests the router has sent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_load: Option<WorkerLoadResponse>,

    /// Job status for async operations (if available).
    pub job_status: Option<JobStatus>,
}

impl WorkerInfo {
    /// Create a partial WorkerInfo for pending workers (not yet registered).
    pub fn pending(worker_id: &str, url: String, job_status: Option<JobStatus>) -> Self {
        Self {
            id: worker_id.to_string(),
            model_id: None,
            spec: Arc::new(WorkerSpec::new(url)),
            is_healthy: false,
            status: Some(WorkerStatus::Pending),
            load: 0,
            http2: false,
            pd_pairing: None,
            stalled: None,
            engine_load: None,
            job_status,
        }
    }
}

/// Job status for async control plane operations
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct JobStatus {
    pub job_type: String,
    pub worker_url: String,
    pub status: String,
    pub message: Option<String>,
    pub timestamp: u64,
}

impl JobStatus {
    /// A status with the given word, stamped now, with the message a
    /// terminal status carries.
    fn stamped(job_type: &str, worker_url: &str, status: &str, message: Option<String>) -> Self {
        Self {
            job_type: job_type.to_string(),
            worker_url: worker_url.to_string(),
            status: status.to_string(),
            message,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Create a pending job status
    pub fn pending(job_type: &str, worker_url: &str) -> Self {
        Self::stamped(job_type, worker_url, "pending", None)
    }

    /// Create a processing job status
    pub fn processing(job_type: &str, worker_url: &str) -> Self {
        Self::stamped(job_type, worker_url, "processing", None)
    }

    /// Create a failed job status
    pub fn failed(job_type: &str, worker_url: &str, error: String) -> Self {
        Self::stamped(job_type, worker_url, "failed", Some(error))
    }
}

#[cfg(test)]
mod job_status_tests {
    use super::JobStatus;

    #[test]
    fn a_status_carries_its_word_its_message_and_a_stamp_of_now() {
        let seconds_since_epoch = || {
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        };
        let before = seconds_since_epoch();
        let pending = JobStatus::pending("AddWorker", "http://w:8000");
        let processing = JobStatus::processing("AddWorker", "http://w:8000");
        let failed = JobStatus::failed("AddWorker", "http://w:8000", "boom".to_string());
        let after = seconds_since_epoch();
        let words = |status: &JobStatus| (status.status.clone(), status.message.clone());
        assert_eq!(words(&pending), ("pending".to_string(), None));
        assert_eq!(words(&processing), ("processing".to_string(), None));
        assert_eq!(
            words(&failed),
            ("failed".to_string(), Some("boom".to_string()))
        );
        for status in [&pending, &processing, &failed] {
            assert_eq!(status.job_type, "AddWorker");
            assert_eq!(status.worker_url, "http://w:8000");
            assert!(
                (before..=after).contains(&status.timestamp),
                "stamped between {before} and {after}, not at {}",
                status.timestamp
            );
        }
    }
}

/// Worker list response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerListResponse {
    pub workers: Vec<WorkerInfo>,
    pub total: usize,
    pub stats: WorkerStats,
}

/// Worker statistics
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerStats {
    pub total_workers: usize,
    pub healthy_workers: usize,
    pub total_models: usize,
    pub total_load: usize,
    pub by_type: WorkerTypeStats,
}

/// Worker statistics by type
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerTypeStats {
    pub regular: usize,
    pub prefill: usize,
    pub decode: usize,
}

// ── Update types ────────────────────────────────────────────────────

/// Partial health check config for PATCH-style updates.
///
/// Each `None` field means "keep the existing value". This avoids the problem
/// where `#[serde(default)]` on [`HealthCheckConfig`] would silently reset
/// unspecified fields to defaults.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HealthCheckUpdate {
    pub timeout_secs: Option<u64>,
    pub check_interval_secs: Option<u64>,
    pub success_threshold: Option<u32>,
    pub failure_threshold: Option<u32>,
    pub disable_health_check: Option<bool>,
    /// Per-worker override for `HealthCheckConfig::drain_settle_secs`.
    pub drain_settle_secs: Option<u64>,
}

impl HealthCheckUpdate {
    /// Returns `true` if all fields are `None` (no overrides specified).
    pub fn is_empty(&self) -> bool {
        self.timeout_secs.is_none()
            && self.check_interval_secs.is_none()
            && self.success_threshold.is_none()
            && self.failure_threshold.is_none()
            && self.disable_health_check.is_none()
            && self.drain_settle_secs.is_none()
    }
}

impl HealthCheckUpdate {
    /// Merge this update into an existing [`HealthCheckConfig`], returning a new config.
    /// Only `Some` fields are applied; `None` fields keep the existing value.
    pub fn apply_to(&self, existing: &HealthCheckConfig) -> HealthCheckConfig {
        HealthCheckConfig {
            timeout_secs: self.timeout_secs.unwrap_or(existing.timeout_secs),
            check_interval_secs: self
                .check_interval_secs
                .unwrap_or(existing.check_interval_secs),
            success_threshold: self.success_threshold.unwrap_or(existing.success_threshold),
            failure_threshold: self.failure_threshold.unwrap_or(existing.failure_threshold),
            disable_health_check: self
                .disable_health_check
                .unwrap_or(existing.disable_health_check),
            drain_settle_secs: self.drain_settle_secs.unwrap_or(existing.drain_settle_secs),
        }
    }
}

/// Per-worker HTTP connection configuration.
/// All fields optional — `None` means "use router/global default".
#[serde_with::skip_serializing_none]
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
pub struct HttpPoolConfig {
    /// Max idle connections per host (default: 500).
    pub pool_max_idle_per_host: Option<usize>,
    /// Idle connection timeout in seconds (default: the router's
    /// `upstream_pool_idle_timeout_secs`).
    pub pool_idle_timeout_secs: Option<u64>,
    /// Request timeout in seconds (default: the router's `request_timeout_secs`).
    pub timeout_secs: Option<u64>,
    /// Connect timeout in seconds (default: 10).
    pub connect_timeout_secs: Option<u64>,
    /// Speak HTTP/2 to this worker via prior knowledge (h2c). `None` lets the
    /// router negotiate per worker when `upstream_http2` is set.
    pub http2: Option<bool>,
}

impl HttpPoolConfig {
    /// Returns `true` if all fields are `None` (no overrides).
    pub fn is_empty(&self) -> bool {
        self.pool_max_idle_per_host.is_none()
            && self.pool_idle_timeout_secs.is_none()
            && self.timeout_secs.is_none()
            && self.connect_timeout_secs.is_none()
            && self.http2.is_none()
    }
}

/// Per-worker absolute overload threshold overrides.
/// All fields optional — `None` means "use gateway default". Either field set
/// enables overload protection for this worker even when the gateway leaves
/// it off. Mirrors `HealthCheckUpdate` pattern for PATCH-style config.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct OverloadUpdate {
    /// Queued (waiting) requests summed across DP ranks at or above which this
    /// worker is considered overloaded. Must be `>= 1`.
    pub waiting_requests: Option<usize>,
    /// Mean KV-cache token usage across DP ranks at or above which this worker
    /// is considered overloaded. Must be in `(0.0, 1.0]`.
    pub token_usage: Option<f64>,
}

impl OverloadUpdate {
    /// Returns `true` if all fields are `None` (no overrides specified).
    pub fn is_empty(&self) -> bool {
        self.waiting_requests.is_none() && self.token_usage.is_none()
    }
}

/// Per-worker resilience overrides (retry + circuit breaker).
/// All fields optional — `None` means "use router default".
/// Mirrors `HealthCheckUpdate` pattern for PATCH-style config.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ResilienceUpdate {
    // ── Retry overrides ──
    /// Max retry attempts (includes first attempt). 1 = no retries.
    pub max_retries: Option<u32>,
    /// Initial backoff delay in milliseconds.
    pub initial_backoff_ms: Option<u64>,
    /// Maximum backoff delay in milliseconds.
    pub max_backoff_ms: Option<u64>,
    /// Backoff multiplier for exponential backoff.
    pub backoff_multiplier: Option<f32>,
    /// Jitter factor (0.0–1.0) applied to backoff delay.
    pub jitter_factor: Option<f32>,
    /// Disable retries entirely for this worker.
    pub disable_retry: Option<bool>,

    // ── Circuit breaker overrides ──
    /// Consecutive failures to open the circuit.
    pub cb_failure_threshold: Option<u32>,
    /// Consecutive successes to close the circuit from half-open.
    pub cb_success_threshold: Option<u32>,
    /// Seconds to wait before attempting half-open.
    pub cb_timeout_secs: Option<u64>,
    /// A closed circuit's failure more than this many seconds after the last
    /// one starts a new streak.
    pub cb_window_secs: Option<u64>,
    /// `true` gives this worker the failure threshold `u32::MAX`, as the
    /// router's `--disable-circuit-breaker` gives every worker, and wins over
    /// `cb_failure_threshold`. `false` under the router's flag gives a
    /// working breaker only together with `cb_failure_threshold`.
    pub disable_circuit_breaker: Option<bool>,

    // ── Retryable status codes ──
    /// HTTP status codes this worker counts as circuit-breaker failures.
    /// When set, replaces the default set (408, 429, 500, 502, 503, 504)
    /// verbatim - entries are not merged in. This does not gate retries:
    /// whether a response is retried is a router-global rule, independent of
    /// this set, so narrowing it cannot make a status non-retryable.
    pub retryable_status_codes: Option<Vec<u16>>,
    /// Capacity-pushback HTTP status codes: still retryable on another
    /// worker, but never counted as circuit-breaker failures (backpressure
    /// is a routing signal, not a fault). When set, replaces the default
    /// set (429).
    pub capacity_status_codes: Option<Vec<u16>>,
}

impl ResilienceUpdate {
    /// Returns `true` if all fields are `None` (no overrides).
    pub fn is_empty(&self) -> bool {
        self.max_retries.is_none()
            && self.initial_backoff_ms.is_none()
            && self.max_backoff_ms.is_none()
            && self.backoff_multiplier.is_none()
            && self.jitter_factor.is_none()
            && self.disable_retry.is_none()
            && self.cb_failure_threshold.is_none()
            && self.cb_success_threshold.is_none()
            && self.cb_timeout_secs.is_none()
            && self.cb_window_secs.is_none()
            && self.disable_circuit_breaker.is_none()
            && self.retryable_status_codes.is_none()
            && self.capacity_status_codes.is_none()
    }
}

/// Worker update request
#[serde_with::skip_serializing_none]
#[derive(Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerUpdateRequest {
    /// Update priority
    pub priority: Option<u32>,

    /// Update cost
    pub cost: Option<f32>,

    /// Update labels
    pub labels: Option<HashMap<String, String>>,

    /// Update API key (for key rotation)
    pub api_key: Option<String>,

    /// Update health check configuration (partial — only specified fields change)
    pub health: Option<HealthCheckUpdate>,
}

impl std::fmt::Debug for WorkerUpdateRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            priority,
            cost,
            labels,
            api_key,
            health,
        } = self;

        f.debug_struct("WorkerUpdateRequest")
            .field("priority", priority)
            .field("cost", cost)
            .field("labels", labels)
            .field("api_key", &api_key.is_some().then_some(REDACTED_API_KEY))
            .field("health", health)
            .finish()
    }
}

// ── Request types ───────────────────────────────────────────────────

/// Query parameters for `GET /workers`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ListWorkersQuery {
    /// Only return workers serving this model, e.g. `?model=moonshotai/Kimi-K2.5`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The shape of the answer: the worker records when absent, or
    /// `manifest`, the file discovery provider's manifest, which a second
    /// gateway reads with `--discovery-provider file`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<WorkerListFormat>,
    /// Only the workers of this origin: `local`, registered at start-up or
    /// through this API, or `discovery`, registered by a discovery provider
    /// that re-registers them on its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<WorkerListOrigin>,
}

/// The shape `GET /workers` answers in, when not the worker records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum WorkerListFormat {
    /// The file discovery provider's manifest: `{"version": 1, "workers": [...]}`.
    Manifest,
}

/// Who registered a worker, as `GET /workers?origin=` filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum WorkerListOrigin {
    /// The start-up flags or the worker API.
    Local,
    /// A discovery provider: Kubernetes, a file, Slurm or Consul.
    Discovery,
}

// ── Response types ──────────────────────────────────────────────────

/// Generic API response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerApiResponse {
    pub success: bool,
    pub message: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerInfo>,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerErrorResponse {
    pub error: String,
    pub code: String,
}

/// Result from flush cache operations across workers
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FlushCacheResult {
    pub successful: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub total_workers: usize,
    pub http_workers: usize,
    /// gRPC workers, and direct-ZMQ engines whose wire carries the flush.
    #[serde(default)]
    pub grpc_workers: usize,
    /// Workers skipped because their transport has no cache-flush RPC
    /// (direct-ZMQ engines whose wire carries no control calls). Keeps
    /// `total = http + grpc + zmq` exact.
    #[serde(default)]
    pub zmq_workers: usize,
    pub message: String,
}

/// Options for starting a profiling run on workers.
///
/// Mirrors the engines' native profile parameters: serialized verbatim as
/// the JSON body for HTTP workers and mapped to the `StartProfile` RPC for
/// gRPC workers. Unset fields fall back to backend defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct ProfileOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_step: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_steps: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub with_stack: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_shapes: Option<bool>,
    pub profile_by_stage: bool,
}

/// Result from profile start/stop operations across workers
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ProfileResult {
    pub successful: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub total_workers: usize,
    pub message: String,
}

/// Request body for the gateway `/start_profile` route: profile options
/// plus an optional worker URL to target a single worker (e.g. one
/// PD-disaggregation role). All workers are profiled when `url` is unset.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct StartProfileRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(flatten)]
    pub options: ProfileOptions,
}

/// Request body for the gateway `/stop_profile` route: optional worker URL
/// to target a single worker.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct StopProfileRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// A token count offered by name from the fixed used-token field.
pub const KV_TOKENS_USED: &str = "kv_tokens_used";
/// A token count offered by name from the fixed token-capacity field.
pub const KV_TOKENS_CAPACITY: &str = "kv_tokens_capacity";

/// The two names offered from the fixed token counts and their unit, each
/// made once.
static FIXED_COUNT_NAMES: LazyLock<[Arc<str>; 3]> = LazyLock::new(|| {
    [
        Arc::from(KV_TOKENS_USED),
        Arc::from(KV_TOKENS_CAPACITY),
        Arc::from("tokens"),
    ]
});

/// One value an engine reports by name alongside its fixed fields. The name
/// and the unit are shared strings: a copy of an entry, which the load
/// monitor makes where several workers share one polled answer and where
/// it publishes a pushed report, allocates none. A connection that knows a
/// name ahead of its reports makes it once and clones it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NamedValue {
    pub name: Arc<str>,
    pub unit: Arc<str>,
    pub value: f64,
    /// Age of this value's sample in milliseconds, when the engine reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age_ms: Option<u32>,
}

/// The scheduler's load report for one data-parallel rank.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct SchedulerLoadSnapshot {
    /// URL of the worker this rank belongs to. Set by a gateway serving a
    /// fleet-wide `/loads`; engines reporting their own load leave it unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    /// `regular`, `prefill`, `decode`, or `encode`. Set alongside `worker`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_type: Option<String>,
    pub dp_rank: i32,
    pub num_running_reqs: i32,
    pub num_waiting_reqs: i32,
    /// Queued token-work: waiting-queue tokens not yet served from cache. 0 when
    /// the backend does not report it — callers degrade gracefully.
    pub num_waiting_uncached_tokens: i32,
    pub num_total_reqs: i32,
    pub num_used_tokens: i32,
    pub max_total_num_tokens: i32,
    /// Token usage ratio (0.0–1.0).
    pub token_usage: f64,
    /// Non-evictable KV occupancy, excluding retained, reclaimable prefix cache.
    /// Older backends omit this; pressure then uses their legacy token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_token_usage: Option<f64>,
    /// Context tokens (prompt plus committed output) of every request the
    /// rank holds. Older backends omit this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_resident_tokens: Option<i64>,
    pub gen_throughput: f64,
    /// Actual non-cached input tokens per active prefill second; unknown/stale is None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_throughput: Option<f64>,
    pub cache_hit_rate: f64,
    pub utilization: f64,
    pub max_running_requests: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<EngineMemoryMetricsSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queues: Option<EngineQueueMetricsSnapshot>,
    /// PD disaggregation signals, populated only when the backend reports a
    /// `disagg` section. `None` for HTTP or older engines. Canonical schema
    /// other engines map into; one engine derives the queue depths from its
    /// per-stage DisaggregationMetrics counters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_speed_gb_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_queue_reqs: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_queue_reqs: Option<i32>,
    /// "prefill", "decode", or "null" as reported by the backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disagg_mode: Option<String>,
    /// Values the engine reports by name, as the one record or answer this
    /// snapshot was read from states them. A pushed record's list takes
    /// the place of the rank's last: a name the record leaves out is not
    /// in its snapshot, and that says nothing about the value. A reader
    /// that wants a value across records keeps the last one it read; the
    /// load sink hears every record.
    ///
    /// A report holds a list to bounds ([`Self::bound_values`]). A pushed
    /// record's snapshot is built under them, and the load monitor puts a
    /// polled answer's list under them before a sink or a reader sees it.
    /// Off the wire the key fails no answer: an entry that is no named
    /// value is read with no name, and a key that is no list as no list.
    #[serde(
        default,
        deserialize_with = "value_bounds::stated",
        skip_serializing_if = "Vec::is_empty"
    )]
    #[schemars(
        description = "Values the engine reports by name, as the one record or answer this \
                       snapshot was read from states them. A list takes the place of the \
                       rank's last: a name it leaves out is not in the snapshot, and that says \
                       nothing about the value. A report keeps the first 64 entries of a list, \
                       less each one with an empty name, a name over 63 bytes, a unit over 31 \
                       bytes, a value that is not finite, or a name an entry kept ahead of it \
                       states."
    )]
    pub values: Vec<NamedValue>,
    /// The parts of this entry that its own source did not carry. A pushed
    /// record's are the parts it left out ([`LoadRecord::snapshot`]): the
    /// load monitor kept them from an older record of the rank, or the
    /// rank has none of them yet. Empty for a polled answer, which has no
    /// key that says a part was left out. Never serialized: an answer on
    /// the wire states its figures and no mark.
    #[serde(skip)]
    #[schemars(skip)]
    pub kept: LoadTelemetry,
    /// How old the kept `num_used_tokens` is at this entry's own sample, in
    /// milliseconds: the time since the count was last stated, by a record
    /// of the rank or by a poll's answer. The time is rounded up to a whole
    /// millisecond: a reader ends the count by this age, and a count must
    /// never read younger than it is. The load monitor writes it, for a
    /// count this entry keeps ([`Self::kept`]), in a report of one rank
    /// and of several: the age counts back from this entry's own sample
    /// ([`Self::sampled_at`]). `None` for a count this entry's own source
    /// states and for a count nothing has stated. Never serialized, as
    /// the mark is not.
    #[serde(skip)]
    #[schemars(skip)]
    pub kept_used_tokens_age_ms: Option<u32>,
    /// How old the kept `max_total_num_tokens` is at this entry's own
    /// sample, as [`Self::kept_used_tokens_age_ms`] says of the used count.
    #[serde(skip)]
    #[schemars(skip)]
    pub kept_total_tokens_age_ms: Option<u32>,
    /// When this rank's state was sampled, on this process's clock, where
    /// the entry's writer knows it apart from its report's
    /// ([`WorkerLoadResponse::sampled_at`]). The load monitor states it
    /// for each row of a worker that pushes its load: a record's receipt
    /// less its age and the one-way margin, or the instant the poll that
    /// laid the row is dated at. A report of several ranks is dated by its
    /// last record of any rank, so a reader that dates one rank's values
    /// takes this instant. `None`: the entry is as old as its report, as
    /// every entry of a polled answer is. Not serialized: it is meaningful
    /// only in the process that set it.
    #[serde(skip)]
    #[schemars(skip)]
    pub sampled_at: Option<std::time::Instant>,
}

impl SchedulerLoadSnapshot {
    /// Puts [`Self::values`] under the bounds a report holds named values
    /// to, in place, and answers how many entries left the list. A report
    /// considers a list's first 64 entries. Of those it drops each one
    /// with an empty name, a name over 63 bytes, a unit over 31 bytes, a
    /// value that is not finite, or a name an entry kept ahead of it
    /// states: the first of a name stands. The kept entries keep their
    /// order.
    ///
    /// The load monitor passes every polled answer through this and counts
    /// what it answers. A pushed record's snapshot is built under the same
    /// bounds. A host that takes an answer by a way of its own calls this
    /// before it reads the list.
    pub fn bound_values(&mut self) -> usize {
        value_bounds::bound(&mut self.values)
    }

    /// Values the engine reports by name, followed by the two fixed token
    /// counts, each where a record stated it and its name is absent from
    /// the list. A stated count is offered at zero too: an engine that goes
    /// idle states zero tokens held, and a reader that keeps the last value
    /// it read must hear it.
    ///
    /// Three rules say what is offered, and all are read here alone. A
    /// count this snapshot's own source stated is offered with no age of
    /// its own. A count the source did not carry ([`Self::kept`]) is
    /// offered where the entry states how old it is
    /// ([`Self::kept_used_tokens_age_ms`]), under that age: the count as
    /// it was last stated, which a reader ends by the age. A
    /// kept count with no age is not offered: no record of the rank has
    /// stated it, and the entry stores a zero. An entry with no token
    /// capacity offers neither count,
    /// whatever its source: an engine that reports its counts reports its
    /// capacity ([`WorkerLoadResponse::has_absolute_token_data`]), and one
    /// that keeps no such size fills both with zero on every record. The
    /// capacity is the entry's, so a record that leaves it out is read by
    /// the one the entry keeps.
    pub fn named_values(&self) -> impl Iterator<Item = NamedValue> + '_ {
        let [used, capacity, tokens] = &*FIXED_COUNT_NAMES;
        let sized = self.max_total_num_tokens > 0;
        let fixed = [
            (
                used,
                self.num_used_tokens,
                LoadTelemetry::USED_TOKENS,
                self.kept_used_tokens_age_ms,
            ),
            (
                capacity,
                self.max_total_num_tokens,
                LoadTelemetry::TOTAL_TOKENS,
                self.kept_total_tokens_age_ms,
            ),
        ]
        .into_iter()
        .filter(move |&(name, _, part, age)| {
            sized
                && (!self.kept.contains(part) || age.is_some())
                && !self.values.iter().any(|value| value.name == *name)
        })
        .map(move |(name, count, part, age)| NamedValue {
            name: Arc::clone(name),
            unit: Arc::clone(tokens),
            value: f64::from(count),
            age_ms: age.filter(|_| self.kept.contains(part)),
        });
        self.values.iter().cloned().chain(fixed)
    }
}

bitflags! {
    /// The telemetry a pushed load record can carry beside its core: one
    /// flag for each part of [`SchedulerLoadSnapshot`] whose figure a report
    /// keeps from the last record that carried it. An optional figure
    /// outside these six is published as the record sends it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct LoadTelemetry: u8 {
        /// `cache_hit_rate`.
        const CACHE_HIT_RATE = 1 << 0;
        /// `num_used_tokens`.
        const USED_TOKENS = 1 << 1;
        /// `max_total_num_tokens`.
        const TOTAL_TOKENS = 1 << 2;
        /// `memory`.
        const MEMORY = 1 << 3;
        /// `queues`.
        const QUEUES = 1 << 4;
        /// The KV transfer latency and speed, the prefill and decode queue
        /// depths and the mode.
        const DISAGGREGATION = 1 << 5;
    }
}

/// A load record pushed on a worker's stream, as the worker's load report
/// and its liveness read it. The wire's message implements this where it is
/// declared, so the code that keeps the report names no wire type.
pub trait LoadRecord {
    /// The sample's age when the record was sent.
    fn age(&self) -> Duration;

    /// The telemetry the record carries. A flagged part it leaves out
    /// keeps, in the report, the figure of the last record that carried it;
    /// nothing else is kept.
    fn telemetry(&self) -> LoadTelemetry;

    /// The record's fields in the shape of a rank's entry, with omitted
    /// telemetry at the type's default and no mark of what was left out.
    /// What a record type writes. A reader takes [`Self::snapshot`].
    fn fields(&self) -> SchedulerLoadSnapshot;

    /// The rank's entry from the record: its fields, marked with the parts
    /// the record did not carry ([`SchedulerLoadSnapshot::kept`]). The
    /// mark is written here, once, for every record type. The rank is the
    /// caller's to set.
    fn snapshot(&self) -> SchedulerLoadSnapshot {
        let mut snapshot = self.fields();
        snapshot.kept = LoadTelemetry::all().difference(self.telemetry());
        snapshot
    }

    /// How many values the record names, ahead of the bounds its snapshot
    /// keeps them under: what the snapshot's list holds fewer was dropped
    /// at a bound, and the load monitor counts it. Zero for a record that
    /// names none. Every record type says it: a default would let one that
    /// carries values count no drop.
    fn values_sent(&self) -> usize;

    /// How long the engine has produced no output while it held requests
    /// it reports on every step. `None` while it holds none, or when the
    /// sender cannot tell.
    fn engine_silence(&self) -> Option<Duration>;

    /// Whether the engine reports every scheduler step, so its silence is
    /// one step's at most while it serves.
    fn reports_every_step(&self) -> bool;

    /// Prompt tokens handed to the engine that have produced no output yet,
    /// across every client of the sender: the prefill work in hand. `None`
    /// when the sender does not report it.
    fn prefill_pending(&self) -> Option<u64>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct EngineMemoryMetricsSnapshot {
    pub weight_gb: f64,
    pub kv_cache_gb: f64,
    pub graph_gb: f64,
    pub token_capacity: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct EngineQueueMetricsSnapshot {
    pub waiting: i32,
    pub grammar: i32,
    pub paused: i32,
    pub retracted: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct EngineAggregateMetricsSnapshot {
    pub total_running_reqs: i32,
    pub total_waiting_reqs: i32,
    pub total_reqs: i32,
    pub avg_token_usage: f64,
    pub avg_throughput: f64,
    pub avg_utilization: f64,
}

impl EngineAggregateMetricsSnapshot {
    /// Roll per-rank snapshots up into one summary. Counts sum, ratios
    /// average, rounded the way the engine servicers round so a gateway
    /// aggregate and an engine aggregate read the same.
    ///
    /// Returns `None` for an empty slice: no ranks reported means no load
    /// data, which must not read as an idle fleet.
    pub fn from_ranks(loads: &[SchedulerLoadSnapshot]) -> Option<Self> {
        if loads.is_empty() {
            return None;
        }
        let n = loads.len() as f64;
        let sum_i32 = |f: fn(&SchedulerLoadSnapshot) -> i32| {
            loads.iter().map(f).fold(0i32, i32::saturating_add)
        };
        let avg = |f: fn(&SchedulerLoadSnapshot) -> f64, decimals: i32| {
            let scale = 10f64.powi(decimals);
            (loads.iter().map(f).sum::<f64>() / n * scale).round() / scale
        };

        let total_running_reqs = sum_i32(|l| l.num_running_reqs);
        let total_waiting_reqs = sum_i32(|l| l.num_waiting_reqs);
        Some(Self {
            total_running_reqs,
            total_waiting_reqs,
            total_reqs: total_running_reqs.saturating_add(total_waiting_reqs),
            avg_token_usage: avg(|l| l.token_usage, 4),
            avg_throughput: avg(|l| l.gen_throughput, 2),
            avg_utilization: avg(|l| l.utilization, 4),
        })
    }
}

/// Full load response for a single worker across all DP ranks.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct WorkerLoadResponse {
    pub timestamp: String,
    pub version: String,
    pub dp_rank_count: i32,
    pub loads: Vec<SchedulerLoadSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregate: Option<EngineAggregateMetricsSnapshot>,
    /// When the engine's state behind this report was sampled, on the
    /// gateway's clock: the poll's receipt, or a pushed record's receipt
    /// less its age and the one-way latency. A policy that books in-flight
    /// work locally releases only what it dispatched before this instant.
    /// Not serialized: it is meaningful only in the process that set it.
    #[serde(skip)]
    #[schemars(skip)]
    pub sampled_at: Option<std::time::Instant>,
}

impl WorkerLoadResponse {
    /// Scope a backend response to one virtual DP worker. A missing rank is
    /// unavailable; it must never inherit another rank or an aggregate. A
    /// gateway fleet rollup cannot be projected because its rank IDs repeat
    /// independently under each annotated worker.
    pub fn for_dp_rank(mut self, rank: Option<usize>) -> Option<Self> {
        if rank.is_some() && !self.ranks_are_dp_ranks() {
            return None;
        }
        if let Some(rank) = rank {
            let rank = i32::try_from(rank).ok()?;
            self.loads.retain(|load| load.dp_rank == rank);
            self.aggregate = None;
            self.dp_rank_count = i32::try_from(self.loads.len()).ok()?;
        }
        if self.loads.is_empty() {
            return None;
        }
        Some(self)
    }

    /// Average token usage ratio across DP ranks. Returns 0.0 if empty.
    pub fn effective_token_usage(&self) -> f64 {
        if self.loads.is_empty() {
            return 0.0;
        }
        self.loads.iter().map(|l| l.token_usage).sum::<f64>() / self.loads.len() as f64
    }

    /// Average non-evictable KV pressure, with the legacy fallback per rank.
    pub fn effective_kv_pressure(&self) -> f64 {
        if self.loads.is_empty() {
            return 0.0;
        }
        self.loads
            .iter()
            .map(|load| load.active_token_usage.unwrap_or(load.token_usage))
            .sum::<f64>()
            / self.loads.len() as f64
    }

    /// Total used tokens summed across all DP ranks.
    pub fn total_used_tokens(&self) -> i64 {
        self.loads.iter().map(|l| l.num_used_tokens as i64).sum()
    }

    /// Whether this response carries real absolute per-rank token counts
    /// (as opposed to a ratio-only snapshot synthesized from Prometheus
    /// `/metrics`, which knows KV *usage* but not token capacity).
    ///
    /// A running engine always reports its KV token capacity, so a positive
    /// `max_total_num_tokens` on any rank marks the absolute-token fields
    /// (`num_used_tokens`, `dp_rank_loads`, `total_used_tokens`) as
    /// meaningful. Callers that need absolute tokens — the DP-rank load
    /// cache — should gate on this so a ratio-only snapshot is not read as
    /// "0 tokens used", and on [`Self::ranks_are_dp_ranks`] so a fleet
    /// rollup is not read as one engine's ranks.
    pub fn has_absolute_token_data(&self) -> bool {
        self.loads.iter().any(|l| l.max_total_num_tokens > 0)
    }

    /// Total queued (waiting, uncached) tokens summed across all DP ranks.
    pub fn total_waiting_uncached_tokens(&self) -> i64 {
        self.loads
            .iter()
            .map(|l| l.num_waiting_uncached_tokens as i64)
            .sum()
    }

    /// Whether the report lists at least one entry and as many as the ranks it
    /// states: a report short of a rank states no whole-worker sum.
    pub fn lists_every_rank(&self) -> bool {
        !self.loads.is_empty()
            && self.loads.len() >= usize::try_from(self.dp_rank_count).unwrap_or(0)
    }

    /// Resident context tokens summed across all DP ranks; `None` unless
    /// every rank reports them. A report that lists fewer entries than the
    /// ranks it states answers `None` too, ahead of any sum: the ranks it
    /// leaves out are in none.
    pub fn total_resident_tokens(&self) -> Option<i64> {
        if !self.lists_every_rank() {
            return None;
        }
        self.loads.iter().map(|l| l.num_resident_tokens).sum()
    }

    /// Total running requests summed across all DP ranks.
    pub fn total_running_reqs(&self) -> i64 {
        self.loads.iter().map(|l| l.num_running_reqs as i64).sum()
    }

    /// Total waiting (queued) requests summed across all DP ranks.
    pub fn total_waiting_reqs(&self) -> i64 {
        self.loads.iter().map(|l| l.num_waiting_reqs as i64).sum()
    }

    /// Total generation throughput (tokens/s) summed across all DP ranks.
    pub fn total_gen_throughput(&self) -> f64 {
        self.loads.iter().map(|l| l.gen_throughput).sum()
    }

    /// Whether these ranks are one engine's DP ranks rather than a
    /// gateway's fleet rollup, where every entry names its own `worker`.
    ///
    /// `dp_rank` is only unique within an engine, so a rollup repeats
    /// `dp_rank: 0` once per non-DP worker. Callers that key by rank — the
    /// DP-rank load cache — must gate on this, or one worker's entry
    /// overwrites another's.
    pub fn ranks_are_dp_ranks(&self) -> bool {
        self.loads.iter().all(|l| l.worker.is_none())
    }

    pub fn dp_rank_loads(&self) -> HashMap<isize, isize> {
        let mut map = HashMap::new();
        for snapshot in &self.loads {
            map.insert(snapshot.dp_rank as isize, snapshot.num_used_tokens as isize);
        }
        map
    }
}

#[cfg(feature = "axum")]
impl IntoResponse for FlushCacheResult {
    fn into_response(self) -> Response {
        let status = if self.failed.is_empty() {
            StatusCode::OK
        } else {
            StatusCode::PARTIAL_CONTENT
        };

        let mut body = json!({
            "status": if self.failed.is_empty() { "success" } else { "partial_success" },
            "message": self.message,
            "workers_flushed": self.successful.len(),
            "total_http_workers": self.http_workers,
            "total_grpc_workers": self.grpc_workers,
            "total_zmq_workers_skipped": self.zmq_workers,
            "total_workers": self.total_workers
        });

        if !self.failed.is_empty() {
            body["successful"] = json!(self.successful);
            body["failed"] = json!(self
                .failed
                .into_iter()
                .map(|(url, err)| json!({"worker": url, "error": err}))
                .collect::<Vec<_>>());
        }

        (status, Json(body)).into_response()
    }
}

#[cfg(feature = "axum")]
impl IntoResponse for ProfileResult {
    fn into_response(self) -> Response {
        let status = if self.total_workers == 0 {
            StatusCode::NOT_FOUND
        } else if self.failed.is_empty() {
            StatusCode::OK
        } else {
            StatusCode::PARTIAL_CONTENT
        };

        let status_str = match status {
            StatusCode::OK => "success",
            StatusCode::PARTIAL_CONTENT => "partial_success",
            _ => "error",
        };
        let mut body = json!({
            "status": status_str,
            "message": self.message,
            "workers_profiled": self.successful.len(),
            "total_workers": self.total_workers
        });

        if !self.failed.is_empty() {
            body["successful"] = json!(self.successful);
            body["failed"] = json!(self
                .failed
                .into_iter()
                .map(|(url, err)| json!({"worker": url, "error": err}))
                .collect::<Vec<_>>());
        }

        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod worker_type_tests {
    use super::WorkerType;

    #[test]
    fn as_str_matches_display_serde_and_from_str() {
        for worker_type in [
            WorkerType::Regular,
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
        ] {
            assert_eq!(worker_type.to_string(), worker_type.as_str());
            assert_eq!(
                serde_json::to_value(worker_type).unwrap(),
                worker_type.as_str()
            );
            assert_eq!(worker_type.as_str().parse(), Ok(worker_type));
        }
    }
}

#[cfg(test)]
mod connection_mode_tests {
    use super::ConnectionMode;

    #[test]
    fn as_str_matches_display_and_serde() {
        for mode in [
            ConnectionMode::Http,
            ConnectionMode::Grpc,
            ConnectionMode::Zmq,
        ] {
            assert_eq!(mode.to_string(), mode.as_str());
            assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_str());
        }
    }

    #[test]
    fn from_url_classifies_known_schemes() {
        assert_eq!(
            ConnectionMode::from_url("ipc:///tmp/e"),
            Some(ConnectionMode::Zmq)
        );
        assert_eq!(
            ConnectionMode::from_url("tcp://10.0.0.5:31000"),
            Some(ConnectionMode::Zmq)
        );
        assert_eq!(
            ConnectionMode::from_url("grpc://h:1"),
            Some(ConnectionMode::Grpc)
        );
        assert_eq!(
            ConnectionMode::from_url("grpcs://h:1"),
            Some(ConnectionMode::Grpc)
        );
        assert_eq!(
            ConnectionMode::from_url("http://h:1"),
            Some(ConnectionMode::Http)
        );
        assert_eq!(
            ConnectionMode::from_url("https://h"),
            Some(ConnectionMode::Http)
        );
    }

    #[test]
    fn from_url_returns_none_for_bare_or_unknown() {
        assert_eq!(ConnectionMode::from_url("host:30000"), None);
        assert_eq!(ConnectionMode::from_url("ftp://host"), None);
    }
}

#[cfg(test)]
mod runtime_type_tests {
    use super::RuntimeType;

    #[test]
    fn generic_round_trips_through_str_and_serde() {
        assert_eq!(RuntimeType::Generic.as_str(), "generic");
        assert_eq!(
            "generic".parse::<RuntimeType>().unwrap(),
            RuntimeType::Generic
        );
        assert_eq!(
            serde_json::to_string(&RuntimeType::Generic).unwrap(),
            "\"generic\""
        );
        assert_eq!(
            serde_json::from_str::<RuntimeType>("\"generic\"").unwrap(),
            RuntimeType::Generic
        );
    }

    #[test]
    fn generic_counts_as_specified() {
        assert!(RuntimeType::Generic.is_specified());
    }
}

#[cfg(test)]
mod worker_status_tests {
    use super::WorkerStatus;

    #[test]
    fn test_try_from_u8_draining() {
        assert_eq!(WorkerStatus::try_from_u8(4), Some(WorkerStatus::Draining));
    }

    #[test]
    fn test_try_from_u8_unknown_returns_none() {
        assert_eq!(WorkerStatus::try_from_u8(5), None);
        assert_eq!(WorkerStatus::try_from_u8(255), None);
    }

    #[test]
    fn test_from_u8_draining() {
        assert_eq!(WorkerStatus::from_u8(4), WorkerStatus::Draining);
    }

    #[test]
    fn test_display_draining_is_snake_case() {
        assert_eq!(WorkerStatus::Draining.to_string(), "draining");
    }

    #[test]
    fn test_serde_round_trip_draining() {
        let s = serde_json::to_string(&WorkerStatus::Draining).unwrap();
        assert_eq!(s, "\"draining\"");
        let back: WorkerStatus = serde_json::from_str(&s).unwrap();
        assert_eq!(back, WorkerStatus::Draining);
    }

    #[test]
    fn test_draining_is_not_routable() {
        assert!(!WorkerStatus::Draining.is_routable());
    }

    #[test]
    fn test_only_ready_is_routable() {
        for s in [
            WorkerStatus::Pending,
            WorkerStatus::Ready,
            WorkerStatus::NotReady,
            WorkerStatus::Failed,
            WorkerStatus::Draining,
        ] {
            assert_eq!(s.is_routable(), s == WorkerStatus::Ready, "{s:?}");
        }
    }
}

#[cfg(test)]
mod health_check_drain_settle_tests {
    use super::{HealthCheckConfig, HealthCheckUpdate};

    #[test]
    fn test_default_drain_settle_secs_is_5() {
        assert_eq!(HealthCheckConfig::default().drain_settle_secs, 5);
    }

    #[test]
    fn test_health_check_update_overrides_drain_settle_secs() {
        let base = HealthCheckConfig::default();
        let update = HealthCheckUpdate {
            drain_settle_secs: Some(30),
            ..Default::default()
        };
        let merged = update.apply_to(&base);
        assert_eq!(merged.drain_settle_secs, 30);
    }

    #[test]
    fn test_health_check_update_keeps_default_when_unset() {
        let base = HealthCheckConfig {
            drain_settle_secs: 12,
            ..Default::default()
        };
        let update = HealthCheckUpdate::default();
        assert_eq!(update.apply_to(&base).drain_settle_secs, 12);
    }

    #[test]
    fn test_health_check_update_is_empty_includes_drain_settle_secs() {
        let mut update = HealthCheckUpdate::default();
        assert!(update.is_empty());
        update.drain_settle_secs = Some(3);
        assert!(!update.is_empty());
    }

    #[test]
    fn test_health_check_config_deserialize_omitted_uses_default() {
        // Existing serialized configs without drain_settle_secs must
        // still deserialize with the default value, not fail.
        let json = r#"{
            "timeout_secs": 30,
            "check_interval_secs": 60,
            "success_threshold": 2,
            "failure_threshold": 3
        }"#;
        let cfg: HealthCheckConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.drain_settle_secs, 5);
    }
}

#[cfg(test)]
mod overload_update_tests {
    use serde_json::json;

    use super::{OverloadUpdate, WorkerSpec};

    #[test]
    fn spec_without_block_deserializes_empty_and_is_not_serialized() {
        // Existing serialized specs carry no `overload` key; they must keep
        // deserializing, and an empty block must not appear on output.
        let spec: WorkerSpec = serde_json::from_value(json!({"url": "http://w:1"})).unwrap();
        assert!(spec.overload.is_empty());

        let out = serde_json::to_value(&spec).unwrap();
        assert!(out.get("overload").is_none());
    }

    #[test]
    fn overload_block_round_trips_per_field() {
        let spec: WorkerSpec = serde_json::from_value(json!({
            "url": "http://w:1",
            "overload": {"waiting_requests": 8, "token_usage": 0.85},
        }))
        .unwrap();
        assert_eq!(spec.overload.waiting_requests, Some(8));
        assert_eq!(spec.overload.token_usage, Some(0.85));

        let out = serde_json::to_value(&spec).unwrap();
        assert_eq!(out["overload"]["waiting_requests"], 8);
        assert_eq!(out["overload"]["token_usage"], 0.85);

        // A one-field block leaves the other signal unset, not defaulted.
        let partial: WorkerSpec = serde_json::from_value(json!({
            "url": "http://w:1",
            "overload": {"token_usage": 0.9},
        }))
        .unwrap();
        assert_eq!(partial.overload.waiting_requests, None);
        assert_eq!(partial.overload.token_usage, Some(0.9));
        assert!(!partial.overload.is_empty());
    }

    #[test]
    fn is_empty_tracks_both_fields() {
        let mut update = OverloadUpdate::default();
        assert!(update.is_empty());
        update.waiting_requests = Some(1);
        assert!(!update.is_empty());
        update = OverloadUpdate {
            waiting_requests: None,
            token_usage: Some(1.0),
        };
        assert!(!update.is_empty());
    }
}

#[cfg(test)]
mod load_snapshot_tests {
    use std::{fs, path::Path};

    use super::*;

    #[test]
    fn named_values_offer_the_fixed_token_counts_independently() {
        // A record that stated neither count.
        let mut snapshot = SchedulerLoadSnapshot {
            kept: LoadTelemetry::USED_TOKENS | LoadTelemetry::TOTAL_TOKENS,
            ..Default::default()
        };
        assert!(snapshot.named_values().next().is_none());

        // A capacity on file, and a record that carried the held count
        // alone.
        snapshot.num_used_tokens = 211;
        snapshot.max_total_num_tokens = 1_373;
        snapshot.kept = LoadTelemetry::TOTAL_TOKENS;
        let used = NamedValue {
            name: "kv_tokens_used".into(),
            unit: "tokens".into(),
            value: 211.0,
            age_ms: None,
        };
        assert_eq!(
            snapshot.named_values().collect::<Vec<_>>().as_slice(),
            std::slice::from_ref(&used)
        );

        snapshot.kept = LoadTelemetry::empty();
        let capacity = NamedValue {
            name: "kv_tokens_capacity".into(),
            unit: "tokens".into(),
            value: 1_373.0,
            age_ms: None,
        };
        let values: Vec<_> = snapshot.named_values().collect();
        assert_eq!(values, [used, capacity.clone()]);
        // The offered names and their unit are made once: two offers share
        // them.
        let again: Vec<_> = snapshot.named_values().collect();
        assert!(values.iter().zip(&again).all(|(first, second)| {
            Arc::ptr_eq(&first.name, &second.name) && Arc::ptr_eq(&first.unit, &second.unit)
        }));
        assert!(Arc::ptr_eq(&values[0].unit, &values[1].unit));

        // A stated count is offered at zero: an idle engine holds none.
        snapshot.num_used_tokens = 0;
        let idle = NamedValue {
            value: 0.0,
            ..values[0].clone()
        };
        assert_eq!(
            snapshot.named_values().collect::<Vec<_>>(),
            [idle, capacity]
        );
        // An entry with no capacity offers neither count, whatever its
        // mark says.
        snapshot.max_total_num_tokens = 0;
        assert!(snapshot.named_values().next().is_none());
    }

    #[test]
    fn the_offer_reads_the_mark_and_the_entrys_capacity() {
        let names = |snapshot: &SchedulerLoadSnapshot| -> Vec<String> {
            snapshot
                .named_values()
                .map(|value| value.name.to_string())
                .collect()
        };
        let entry = |capacity: i32, kept: LoadTelemetry| SchedulerLoadSnapshot {
            num_used_tokens: 211,
            max_total_num_tokens: capacity,
            kept,
            ..Default::default()
        };
        // A capacity: each count the source carried.
        assert_eq!(
            names(&entry(1_373, LoadTelemetry::empty())),
            [KV_TOKENS_USED, KV_TOKENS_CAPACITY]
        );
        assert_eq!(
            names(&entry(1_373, LoadTelemetry::TOTAL_TOKENS)),
            [KV_TOKENS_USED]
        );
        // No capacity, zero or negative: neither count, carried or not.
        for capacity in [0, -1] {
            assert!(names(&entry(capacity, LoadTelemetry::empty())).is_empty());
        }
        // The bound of a list leaves the mark as it is.
        let mark = LoadTelemetry::USED_TOKENS | LoadTelemetry::MEMORY;
        let mut bounded = entry(1_373, mark);
        assert_eq!(bounded.bound_values(), 0);
        assert_eq!(bounded.kept, mark);
    }

    #[test]
    fn a_polled_rank_with_no_token_capacity_states_neither_fixed_count() {
        let names = |snapshot: &SchedulerLoadSnapshot| -> Vec<(String, f64)> {
            snapshot
                .named_values()
                .map(|value| (value.name.to_string(), value.value))
                .collect()
        };
        // No capacity: the answer's zero, or a count beside it, is no
        // statement.
        for used in [0, 211] {
            let mut polled = SchedulerLoadSnapshot {
                num_used_tokens: used,
                ..Default::default()
            };
            assert_eq!(polled.bound_values(), 0);
            assert!(names(&polled).is_empty(), "{used} used");
        }
        // A capacity: both counts are the engine's, the zero too.
        let mut polled = SchedulerLoadSnapshot {
            max_total_num_tokens: 1_373,
            ..Default::default()
        };
        assert_eq!(polled.bound_values(), 0);
        assert_eq!(
            names(&polled),
            [
                (KV_TOKENS_USED.to_owned(), 0.0),
                (KV_TOKENS_CAPACITY.to_owned(), 1_373.0)
            ]
        );
    }

    /// An entry's size. A report holds its entries until the worker's next
    /// one, a poll that several workers share copies entries for each of
    /// them but the last (the entries of the worker's own rank where an
    /// engine answers, every entry of a gateway's rollup), and a pushed
    /// report's publish copies each of its entries, so a field the entry
    /// gains is a cost of each, on the path with a policy engine and on the
    /// one without: the pin is exact. The entry's own sample instant is two
    /// words of it.
    #[test]
    fn a_load_entry_takes_the_pinned_bytes() {
        assert_eq!(size_of::<Option<std::time::Instant>>(), 16);
        assert_eq!(size_of::<SchedulerLoadSnapshot>(), 352);
    }

    /// A count the entry keeps from an older record is offered where the
    /// entry states its age, under that age, and not where it states none.
    /// A count of the entry's own record is offered with no age, whatever
    /// age the entry holds for it.
    #[test]
    fn named_values_offer_a_kept_count_under_its_age_and_none_without_one() {
        let mut snapshot = SchedulerLoadSnapshot {
            num_used_tokens: 211,
            max_total_num_tokens: 1_373,
            ..Default::default()
        };
        let offered = |snapshot: &SchedulerLoadSnapshot| -> Vec<(String, f64, Option<u32>)> {
            snapshot
                .named_values()
                .map(|value| (value.name.to_string(), value.value, value.age_ms))
                .collect()
        };
        let used = |age| (KV_TOKENS_USED.to_owned(), 211.0, age);
        let capacity = |age| (KV_TOKENS_CAPACITY.to_owned(), 1_373.0, age);
        assert_eq!(offered(&snapshot), [used(None), capacity(None)]);
        // Kept with no age: not offered, as a record no instant is known of.
        snapshot.kept = LoadTelemetry::USED_TOKENS;
        assert_eq!(offered(&snapshot), [capacity(None)]);
        snapshot.kept = LoadTelemetry::USED_TOKENS | LoadTelemetry::TOTAL_TOKENS;
        assert!(offered(&snapshot).is_empty());
        // Kept with an age: offered under it, each count by its own.
        snapshot.kept_used_tokens_age_ms = Some(40);
        assert_eq!(offered(&snapshot), [used(Some(40))]);
        snapshot.kept_total_tokens_age_ms = Some(0);
        assert_eq!(offered(&snapshot), [used(Some(40)), capacity(Some(0))]);
        // The entry's own record states the used count again: no age.
        snapshot.kept = LoadTelemetry::TOTAL_TOKENS;
        assert_eq!(offered(&snapshot), [used(None), capacity(Some(0))]);
        // A list that states a count's name stands for it, kept or not.
        snapshot.values = vec![NamedValue {
            name: KV_TOKENS_CAPACITY.into(),
            unit: "tokens".into(),
            value: 9.0,
            age_ms: None,
        }];
        assert_eq!(
            offered(&snapshot),
            [(KV_TOKENS_CAPACITY.to_owned(), 9.0, None), used(None)]
        );
        // The mark and the ages are the process's own: the answer on the
        // wire states both counts and neither.
        let json = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(json["num_used_tokens"], 211);
        assert_eq!(json["max_total_num_tokens"], 1_373);
        for key in [
            "kept",
            "kept_used_tokens_age_ms",
            "kept_total_tokens_age_ms",
        ] {
            assert!(json.get(key).is_none(), "{key}");
        }
    }

    #[test]
    fn explicit_token_values_take_precedence_over_fixed_counts() {
        for (name, other) in [
            ("kv_tokens_used", "kv_tokens_capacity"),
            ("kv_tokens_capacity", "kv_tokens_used"),
        ] {
            let value = NamedValue {
                name: name.into(),
                unit: "tokens".into(),
                value: 0.0,
                age_ms: Some(43),
            };
            let snapshot = SchedulerLoadSnapshot {
                num_used_tokens: 313,
                max_total_num_tokens: 2_812,
                values: vec![value.clone()],
                ..Default::default()
            };
            let values: Vec<_> = snapshot.named_values().collect();
            assert_eq!(values.len(), 2);
            assert_eq!(values[0], value);
            assert_eq!(&*values[1].name, other);
        }
    }

    #[test]
    fn the_snapshot_schema_matches_the_snapshot_type() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scheduler-load.schema.json");
        let schema = schemars::schema_for!(SchedulerLoadSnapshot);
        let generated = serde_json::to_string_pretty(&schema).unwrap() + "\n";
        if std::env::var_os("WRITE_SCHEDULER_LOAD_SCHEMA").is_some() {
            fs::write(path, generated).unwrap();
        } else {
            assert_eq!(
                fs::read_to_string(path).unwrap(),
                generated,
                "scheduler-load.schema.json is stale; write it again with \
                 WRITE_SCHEDULER_LOAD_SCHEMA=1 cargo test -p smg-worker --all-features \
                 the_snapshot_schema_matches_the_snapshot_type",
            );
        }
    }
}

#[cfg(test)]
mod aggregate_rollup_tests {
    use super::{EngineAggregateMetricsSnapshot, SchedulerLoadSnapshot};

    fn rank(
        num_running_reqs: i32,
        num_waiting_reqs: i32,
        token_usage: f64,
        gen_throughput: f64,
        utilization: f64,
    ) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot {
            num_running_reqs,
            num_waiting_reqs,
            token_usage,
            gen_throughput,
            utilization,
            ..Default::default()
        }
    }

    #[test]
    fn no_ranks_is_none_not_idle() {
        assert!(EngineAggregateMetricsSnapshot::from_ranks(&[]).is_none());
    }

    #[test]
    fn counts_sum_and_ratios_average() {
        let aggregate = EngineAggregateMetricsSnapshot::from_ranks(&[
            rank(4, 1, 0.2, 100.0, 0.3),
            rank(6, 3, 0.6, 200.0, 0.7),
        ])
        .expect("aggregate");

        assert_eq!(aggregate.total_running_reqs, 10);
        assert_eq!(aggregate.total_waiting_reqs, 4);
        assert_eq!(aggregate.total_reqs, 14);
        assert_eq!(aggregate.avg_token_usage, 0.4);
        assert_eq!(aggregate.avg_throughput, 150.0);
        assert_eq!(aggregate.avg_utilization, 0.5);
    }

    #[test]
    fn averages_round_the_way_the_engines_round() {
        // 1/3 at 4 decimals for ratios, 2 for throughput.
        let aggregate = EngineAggregateMetricsSnapshot::from_ranks(&[
            rank(0, 0, 1.0, 1.0, 1.0),
            rank(0, 0, 0.0, 0.0, 0.0),
            rank(0, 0, 0.0, 0.0, 0.0),
        ])
        .expect("aggregate");

        assert_eq!(aggregate.avg_token_usage, 0.3333);
        assert_eq!(aggregate.avg_throughput, 0.33);
        assert_eq!(aggregate.avg_utilization, 0.3333);
    }

    #[test]
    fn counts_saturate_instead_of_overflowing() {
        let aggregate = EngineAggregateMetricsSnapshot::from_ranks(&[
            rank(i32::MAX, i32::MAX, 0.0, 0.0, 0.0),
            rank(1, 1, 0.0, 0.0, 0.0),
        ])
        .expect("aggregate");

        assert_eq!(aggregate.total_running_reqs, i32::MAX);
        assert_eq!(aggregate.total_reqs, i32::MAX);
    }
}

#[cfg(test)]
mod constrained_decoding_tests {
    use super::*;

    /// A label names one of the two policies or nothing; the spec's typed
    /// value wins over the label; absent everywhere, a worker is `requested`.
    #[test]
    fn constrained_decoding_resolves_spec_then_label() {
        assert_eq!(
            ConstrainedDecoding::parse("always"),
            Ok(ConstrainedDecoding::Always)
        );
        assert_eq!(
            ConstrainedDecoding::parse(" requested "),
            Ok(ConstrainedDecoding::Requested)
        );
        assert!(ConstrainedDecoding::parse("sometimes")
            .unwrap_err()
            .contains("'requested' or 'always'"));
        assert_eq!(
            ConstrainedDecoding::from_label(None),
            ConstrainedDecoding::Requested
        );
        assert_eq!(
            ConstrainedDecoding::from_label(Some("always")),
            ConstrainedDecoding::Always
        );
        assert_eq!(
            ConstrainedDecoding::from_label(Some("sometimes")),
            ConstrainedDecoding::Requested
        );
        assert_eq!(ConstrainedDecoding::Always.as_label(), "always");

        let mut spec = WorkerSpec::new("http://w:8000");
        assert_eq!(
            ConstrainedDecoding::of(&spec),
            ConstrainedDecoding::Requested
        );
        spec.labels
            .insert(ConstrainedDecoding::LABEL.to_string(), "always".to_string());
        assert_eq!(ConstrainedDecoding::of(&spec), ConstrainedDecoding::Always);
        spec.constrained_decoding = Some(ConstrainedDecoding::Requested);
        assert_eq!(
            ConstrainedDecoding::of(&spec),
            ConstrainedDecoding::Requested
        );
    }

    /// The spec field round-trips through JSON in the label's spelling and is
    /// left out when unset.
    #[test]
    fn constrained_decoding_serializes_as_its_label_value() {
        let mut spec = WorkerSpec::new("http://w:8000");
        assert!(!serde_json::to_string(&spec)
            .unwrap()
            .contains("constrained_decoding"));
        spec.constrained_decoding = Some(ConstrainedDecoding::Always);
        let json = serde_json::to_value(&spec).unwrap();
        assert_eq!(json["constrained_decoding"], "always");
        let parsed: WorkerSpec =
            serde_json::from_str(r#"{"url":"http://w:8000","constrained_decoding":"requested"}"#)
                .unwrap();
        assert_eq!(
            parsed.constrained_decoding,
            Some(ConstrainedDecoding::Requested)
        );
        assert!(serde_json::from_str::<WorkerSpec>(
            r#"{"url":"http://w:8000","constrained_decoding":"sometimes"}"#
        )
        .is_err());
    }
}

#[cfg(test)]
mod debug_tests {
    use super::{WorkerSpec, WorkerUpdateRequest};

    #[test]
    fn a_spec_prints_no_api_key() {
        let marker = "pebble";
        let mut spec = WorkerSpec::new("http://worker:8000");
        spec.api_key = Some(marker.to_string());

        let compact = format!("{spec:?}");
        let pretty = format!("{spec:#?}");
        assert!(!compact.contains(marker), "{compact}");
        assert!(!pretty.contains(marker), "{pretty}");
        assert!(
            compact.contains(r#"api_key: Some("<redacted>")"#),
            "{compact}"
        );
        assert!(
            pretty.contains("api_key: Some(\n        \"<redacted>\",\n    )"),
            "{pretty}"
        );

        spec.api_key = None;
        for text in [format!("{spec:?}"), format!("{spec:#?}")] {
            assert!(text.contains("api_key: None"), "{text}");
        }
    }

    #[test]
    fn a_spec_prints_no_credential_of_its_addresses() {
        let mut spec = WorkerSpec::new("http://user:pebble@worker:8000");
        spec.dp_base_url = Some("http://cedar@worker:8001".to_string());

        for text in [format!("{spec:?}"), format!("{spec:#?}")] {
            assert!(!text.contains("pebble"), "{text}");
            assert!(!text.contains("cedar"), "{text}");
            assert!(text.contains("***@worker:8000"), "{text}");
            assert!(text.contains("***@worker:8001"), "{text}");
        }
    }

    #[test]
    fn a_spec_reads_a_rank_out_of_the_address_beside_a_base_alone() {
        // With no base, an `@` and digits at the end are a userinfo and a
        // host.
        let mut spec = WorkerSpec::new("http://pebble@2130706433");
        for text in [format!("{spec:?}"), format!("{spec:#?}")] {
            assert!(!text.contains("pebble"), "{text}");
        }

        // A base carries no rank, and the address built from it does: the
        // base, an `@` and digits.
        spec.url = "http://cedar@worker:8000@1".to_string();
        spec.dp_base_url = Some("http://cedar@worker:8000".to_string());
        for text in [format!("{spec:?}"), format!("{spec:#?}")] {
            assert!(!text.contains("cedar"), "{text}");
            assert!(
                text.contains(r#"url: "http://***@worker:8000@1""#),
                "{text}"
            );
            assert!(text.contains(r#""http://***@worker:8000""#), "{text}");
        }

        // The address built from a base that itself ends in an `@` and
        // digits keeps its rank. The base alone reads as none: its `@`
        // parts a userinfo from a host of digits.
        spec.url = "http://cedar@8001@1".to_string();
        spec.dp_base_url = Some("http://cedar@8001".to_string());
        for text in [format!("{spec:?}"), format!("{spec:#?}")] {
            assert!(!text.contains("cedar"), "{text}");
            assert!(text.contains(r#"url: "http://***@8001@1""#), "{text}");
            assert!(text.contains(r#""http://***@8001""#), "{text}");
        }
    }

    #[test]
    fn a_base_beside_an_address_that_is_not_its_rank_prints_no_userinfo() {
        // A base is set and the address is not that base with a rank: the
        // `@` and the digits at its end are a userinfo and a numeric host.
        let mut spec = WorkerSpec::new("http://pebble@2130706433");
        spec.dp_base_url = Some("http://other:8000".to_string());
        for text in [format!("{spec:?}"), format!("{spec:#?}")] {
            assert!(!text.contains("pebble"), "{text}");
        }
    }

    #[test]
    fn a_spec_prints_a_socket_path_that_ends_like_a_rank_as_it_stands() {
        // An IPC address holds no userinfo: the `@` and the digits at its
        // end are bytes of its path, in an address and in a base.
        let mut spec = WorkerSpec::new("ipc:///tmp/w@0");
        spec.dp_base_url = Some("ipc:///tmp/g@00".to_string());
        assert!(!spec.url_is_ranked());
        assert_eq!(spec.redacted_url(), "ipc:///tmp/w@0");
        for text in [format!("{spec:?}"), format!("{spec:#?}")] {
            assert!(text.contains(r#"url: "ipc:///tmp/w@0""#), "{text}");
            assert!(text.contains(r#""ipc:///tmp/g@00""#), "{text}");
            assert!(!text.contains("unparsable"), "{text}");
        }
    }

    #[test]
    fn a_spec_ranks_the_address_that_is_its_base_and_digits_alone() {
        let mut spec = WorkerSpec::new("http://worker:8000@1");
        // No base: no rank is read, and the `@` parts a userinfo from a
        // host of digits.
        assert!(!spec.url_is_ranked());
        assert_eq!(spec.redacted_url(), "http://***@1");
        spec.dp_base_url = Some("http://worker:8000".to_string());
        assert!(spec.url_is_ranked());
        assert_eq!(spec.redacted_url(), "http://worker:8000@1");
        // The base, an `@` and digits, and nothing else.
        for url in [
            "http://worker:8000",
            "http://worker:8000@",
            "http://worker:8000@1a",
            "http://worker:80001@1",
            "http://other:8000@1",
        ] {
            spec.url = url.to_string();
            assert!(!spec.url_is_ranked(), "{url}");
        }
        // A base with a userinfo is masked, and its rank is kept.
        spec.url = "http://cedar@worker:8000@12".to_string();
        spec.dp_base_url = Some("http://cedar@worker:8000".to_string());
        assert!(spec.url_is_ranked());
        assert_eq!(spec.redacted_url(), "http://***@worker:8000@12");
        // A plain address is the registered string, byte for byte.
        let plain = WorkerSpec::new("http://Worker:8080/");
        assert_eq!(plain.redacted_url(), "http://Worker:8080/");
        // A userinfo before a host of digits is masked.
        let token = WorkerSpec::new("http://pebble@2130706433");
        assert_eq!(token.redacted_url(), "http://***@2130706433");
    }

    #[test]
    fn a_key_rotation_request_prints_no_api_key() {
        let marker = "cedar";
        let mut request = WorkerUpdateRequest {
            priority: Some(73),
            cost: None,
            labels: None,
            api_key: Some(marker.to_string()),
            health: None,
        };

        let compact = format!("{request:?}");
        let pretty = format!("{request:#?}");
        assert!(!compact.contains(marker), "{compact}");
        assert!(!pretty.contains(marker), "{pretty}");
        assert!(
            compact.contains(r#"api_key: Some("<redacted>")"#),
            "{compact}"
        );
        assert!(
            pretty.contains("api_key: Some(\n        \"<redacted>\",\n    )"),
            "{pretty}"
        );

        request.api_key = None;
        let expected = "WorkerUpdateRequest { priority: Some(73), cost: None, labels: None, api_key: None, health: None }";
        assert_eq!(format!("{request:?}"), expected);
        let pretty = format!("{request:#?}");
        assert!(pretty.contains("api_key: None"), "{pretty}");
    }

    #[test]
    fn a_spec_still_prints_its_other_fields() {
        let spec = WorkerSpec {
            priority: 73,
            ..WorkerSpec::new("http://worker:8000")
        };

        let text = format!("{spec:?}");
        assert!(
            text.starts_with(r#"WorkerSpec { url: "http://worker:8000","#),
            "{text}"
        );
        assert!(text.contains("priority: 73"), "{text}");

        let pretty = format!("{spec:#?}");
        let field_count = pretty
            .lines()
            .filter_map(|line| line.strip_prefix("    "))
            .filter(|line| line.starts_with(|c: char| c.is_ascii_lowercase()))
            .count();
        assert_eq!(field_count, 30);
    }
}
