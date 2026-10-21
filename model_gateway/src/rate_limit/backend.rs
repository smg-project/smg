//! `RateLimitBackend`: the reserve/settle/close/abandon contract a rate
//! limit implementation must provide. `Local` (this phase) is in-memory;
//! a future `External` (HTTP-based, distributed) implementation is the
//! reason this is a trait rather than a concrete type.

use async_trait::async_trait;
use uuid::Uuid;

use crate::tenant::TenantKey;

/// What a reservation attempt asks for. `model_id` is optional — when
/// absent, model-rule scoping is skipped.
#[derive(Debug, Clone)]
pub struct ReserveRequest {
    pub request_charge_id: Uuid,
    pub tenant_key: TenantKey,
    pub model_id: Option<String>,
    pub estimated_input_tokens: u32,
}

/// Outcome of a `reserve` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveOutcome {
    Admitted,
    Denied { retry_after_secs: u64 },
}

/// Authoritative usage after a successful call, used to true up the
/// reservation via `(actual_input_tokens + completion_tokens) -
/// estimated_input_tokens`.
///
/// `cached_input_tokens` and `reasoning_tokens` are the engine's own split
/// of those two counts: the prompt tokens it served from its prefix cache
/// and the completion tokens it spent on reasoning. They take no part in
/// the true-up; they are carried for the consumers that price or report a
/// settlement, and are zero where the engine reports none (a reasoning
/// split the gateway's own parser made is not substituted here).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageSettlement {
    pub actual_input_tokens: u32,
    pub completion_tokens: u32,
    /// Of `actual_input_tokens`, the tokens served from the engine's cache.
    pub cached_input_tokens: u32,
    /// Of `completion_tokens`, the tokens the engine counted as reasoning.
    pub reasoning_tokens: u32,
}

/// Reserve-once-per-logical-request, settle/close/abandon-once lifecycle,
/// keyed by `request_charge_id` (not the client-correlatable request id)
/// so retries reuse one reservation.
#[async_trait]
pub trait RateLimitBackend: Send + Sync {
    async fn reserve(&self, request: ReserveRequest) -> ReserveOutcome;

    /// Settle from final, authoritative usage. Idempotent: a no-op if the
    /// reservation was already settled/closed/abandoned.
    async fn settle_success(&self, request_charge_id: Uuid, usage: UsageSettlement);

    /// Close without settlement (success with no usage reported, or a
    /// terminal failure after admission) — keeps the reserved-input debit
    /// rather than refunding it. Idempotent.
    async fn close_reserved_only(&self, request_charge_id: Uuid);

    /// Abandon on drop/cancel/disconnect before an explicit close — same
    /// keep-the-debit semantics as `close_reserved_only`. Idempotent.
    async fn abandon(&self, request_charge_id: Uuid);
}
