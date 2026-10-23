//! A worker's metadata: the protocol-level spec it embeds and the
//! internal-only fields for health checking and endpoint routing, with the
//! readers the registry, the routers and the bindings use.

use std::{collections::HashMap, sync::Arc, time::Duration};

use openai_protocol::{
    model_card::ModelCard,
    model_type::{Endpoint, ModelType},
    worker::{HealthCheckConfig, ProviderType, WorkerSpec},
};

use super::{
    overload::OverloadThresholds, pd_pairing::PdPairing, WorkerError, WorkerResult,
    UNKNOWN_MODEL_ID,
};

/// Metadata associated with a worker.
///
/// Embeds [`WorkerSpec`] for identity/config fields shared with the
/// protocol layer, plus internal-only fields for health checking and
/// endpoint routing.
#[derive(Debug, Clone)]
pub struct WorkerMetadata {
    /// Protocol-level worker identity and configuration.
    ///
    /// Behind an `Arc` so it can be shared into response `WorkerInfo`s (e.g.
    /// `GET /workers`) without a per-worker deep clone. Set once at construction
    /// and not mutated afterwards.
    pub spec: Arc<WorkerSpec>,
    /// Resolved health check config (router defaults + per-worker overrides).
    /// This is the concrete config used at runtime; `spec.health` only stores
    /// the partial overrides from the API layer.
    pub health_config: HealthCheckConfig,
    /// Health check endpoint path (internal-only, from router config).
    pub health_endpoint: String,
    /// The longest any one stage of a ZMQ engine's startup handshake may stay
    /// silent (from router config); it alone covers the engine's init between
    /// its HELLO and its READY, which the gateway sees nothing of otherwise.
    pub zmq_startup_timeout: Duration,
    /// The bound on a request over the worker's ZMQ link, from its dispatch
    /// to its last message (resolved at registration as the HTTP worker
    /// client's per-request timeout is: the worker's own
    /// `http_pool.timeout_secs`, else the router's `request_timeout_secs`);
    /// `None` leaves the link's streams open for as long as the client reads
    /// them.
    pub zmq_request_timeout: Option<Duration>,
    /// Effective absolute overload thresholds (per signal: `spec.overload`
    /// override, else gateway default). Resolved once at registration; the
    /// load monitor's ingestion predicate scores every report against these,
    /// so nothing on a request path re-resolves them.
    pub overload: OverloadThresholds,
    /// Whether the router speaks HTTP/2 prior knowledge to this worker:
    /// `spec.http_pool.http2` when declared, else negotiated at registration
    /// under `upstream_http2`.
    pub http2: bool,
    /// What this worker offers a PD rendezvous partner, derived once from
    /// the spec and its discovered labels (#2483).
    pub pd_pairing: PdPairing,
    /// The labels discovery read from the engine when the worker was
    /// registered or last refreshed, before the operator's own labels were
    /// merged over them. A refresh on the engine's next transition into
    /// service replaces exactly these (see the update step's
    /// `refreshed_labels`), so a label the operator set or overrode keeps its
    /// value. Empty for a worker that was not discovered (a mesh import, a
    /// test fixture).
    pub discovered_labels: Arc<HashMap<String, String>>,
}

impl WorkerMetadata {
    // ── Identity / transport ────────────────────────────────────────

    /// Get the bootstrap hostname for PD mode (parsed from URL at
    /// construction time).
    pub fn bootstrap_host(&self) -> &str {
        &self.spec.bootstrap_host
    }

    /// Get the bootstrap port for PD mode.
    pub fn bootstrap_port(&self) -> Option<u16> {
        self.spec.bootstrap_port
    }

    /// Get the base URL without any DP rank suffix.
    pub fn base_url(&self) -> &str {
        self.spec
            .dp_base_url
            .as_deref()
            .unwrap_or(self.spec.url.as_str())
    }

    /// Compose an endpoint URL for a specific route.
    pub fn endpoint_url(&self, route: &str) -> String {
        format!("{}{}", self.base_url(), route)
    }

    /// Whether the router speaks HTTP/2 prior knowledge to this worker.
    pub fn http2(&self) -> bool {
        self.http2
    }

    // ── DP awareness ────────────────────────────────────────────────

    /// Check if this worker is DP-aware.
    pub fn is_dp_aware(&self) -> bool {
        self.spec.dp_rank.is_some()
    }

    /// Get DP rank if this is a DP-aware worker.
    pub fn dp_rank(&self) -> Option<usize> {
        self.spec.dp_rank
    }

    /// Get DP size if this worker is part of a DP group.
    pub fn dp_size(&self) -> Option<usize> {
        self.spec.dp_size
    }

    /// Number of ZMQ engines this worker's handshake awaits. A grouped ZMQ
    /// worker carries `dp_size = Some(N)` with no `dp_rank` — one worker, one
    /// socket set, N engines dialing in (the connector balances across them
    /// and drives the wave protocol for lockstep groups). Rank-expanded DP
    /// workers never reach ZMQ connect (rejected at registration).
    pub fn zmq_engine_count(&self) -> usize {
        self.spec.dp_size.unwrap_or(1).max(1)
    }

    /// Transform a request for DP-aware routing.
    ///
    /// When the worker has a `dp_rank`, injects `data_parallel_rank`
    /// into the request body. Otherwise returns the request unchanged.
    /// Sync because the body is pure JSON manipulation — the previous
    /// `async` on the trait method had no `await` inside.
    pub fn prepare_request(&self, mut req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        if let Some(rank) = self.spec.dp_rank {
            if let Some(map) = req.as_object_mut() {
                map.insert("data_parallel_rank".to_string(), serde_json::json!(rank));
                Ok(req)
            } else {
                Err(WorkerError::InvalidConfiguration {
                    message: "Request must be a JSON object for DP-aware routing".to_string(),
                })
            }
        } else {
            Ok(req)
        }
    }

    /// True when [`Self::prepare_request`] would modify the request.
    pub fn mutates_request(&self) -> bool {
        self.spec.dp_rank.is_some()
    }

    // ── Routing priorities / model lookup ───────────────────────────

    /// Get the model ID this worker serves.
    ///
    /// Checks `ModelCards` first, then falls back to the `model_id`
    /// label, and finally [`UNKNOWN_MODEL_ID`] if nothing is set.
    pub fn model_id(&self) -> &str {
        self.spec
            .models
            .primary()
            .map(|m| m.id.as_str())
            .or_else(|| self.spec.labels.get("model_id").map(|s| s.as_str()))
            .unwrap_or(UNKNOWN_MODEL_ID)
    }

    /// Get the priority of this worker (higher value = higher priority).
    pub fn priority(&self) -> u32 {
        self.spec.priority
    }

    /// Get the cost factor of this worker (baseline = 1.0).
    pub fn cost(&self) -> f32 {
        self.spec.cost
    }

    /// Get the default provider type for this worker.
    /// `None` means native/passthrough.
    pub fn default_provider(&self) -> Option<&ProviderType> {
        self.spec.provider.as_ref()
    }

    // ── Model lookups ───────────────────────────────────────────────

    /// Find a model card by ID (including aliases)
    pub fn find_model(&self, model_id: &str) -> Option<&ModelCard> {
        self.spec.models.find(model_id)
    }

    /// Check if this worker can serve a given model.
    /// Wildcard workers accept any model.
    pub fn supports_model(&self, model_id: &str) -> bool {
        self.spec.models.supports(model_id)
    }

    /// Check if this worker supports an endpoint for a given model.
    /// Falls back to LLM capabilities if model not found — this is safe because
    /// non-LLM workers (embeddings, rerank) are always registered with explicit
    /// models via discovery, never as wildcards.
    pub fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        if let Some(model) = self.find_model(model_id) {
            model.supports_endpoint(endpoint)
        } else {
            ModelType::LLM.supports_endpoint(endpoint)
        }
    }

    /// Get the provider for a given model.
    /// Returns the model's provider if found, otherwise the worker's default provider.
    pub fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.find_model(model_id)
            .and_then(|m| m.provider.as_ref())
            .or(self.spec.provider.as_ref())
    }

    /// Get all model IDs this worker can serve
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.spec.models.iter().map(|m| m.id.as_str())
    }

    /// Check if this worker is in wildcard mode (accepts any model).
    pub fn is_wildcard(&self) -> bool {
        self.spec.models.is_wildcard()
    }
}
