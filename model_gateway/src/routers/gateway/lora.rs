//! SGLang's OpenAI `base:adapter` selector, resolved before family routing.
//! Exact registered model names and aliases always take precedence. The
//! request body then carries the base model and the existing LoRA extension
//! through tokenization, cache routing, and the scheduler request builder.

use std::sync::Arc;

use axum::response::Response;

use super::{route_error, router_ids, Gateway, RouterTrait};
use crate::worker::{ConnectionMode, RoutingPool, RuntimeType, Worker, WorkerRegistry};

fn sglang_grpc(worker: &dyn Worker) -> bool {
    worker.metadata().spec.runtime_type == RuntimeType::Sglang
        && worker.connection_mode() == &ConnectionMode::Grpc
}

fn has_sglang_grpc(registry: &WorkerRegistry, model: &str) -> bool {
    registry
        .get_by_model(model)
        .iter()
        .any(|worker| sglang_grpc(worker.as_ref()))
}

pub(super) fn normalize_model(
    registry: &WorkerRegistry,
    model: &mut String,
    lora_path: &mut Option<String>,
) -> Result<Option<String>, Response> {
    let Some((base, adapter)) = model.rsplit_once(':') else {
        return Ok(None);
    };
    // Includes aliases and provider model IDs containing colons.
    if !registry.get_by_model(model).is_empty() {
        return Ok(None);
    }
    if !has_sglang_grpc(registry, base) {
        return Ok(None);
    }
    if adapter.trim().is_empty() {
        return Err(route_error::bad_request(
            "invalid_lora_adapter",
            "model:adapter requires a non-empty adapter name",
        ));
    }
    if lora_path.as_deref().is_some_and(|path| path != adapter) {
        return Err(route_error::bad_request(
            "conflicting_lora_adapter",
            "model adapter and lora_path must select the same adapter",
        ));
    }
    let base = base.to_string();
    *lora_path = Some(adapter.to_string());
    model.truncate(base.len());
    Ok(Some(base))
}

impl Gateway {
    /// Preserve HTTP-only native contracts. When the model has a SGLang
    /// gRPC pool, an explicit adapter must not fall through to a family
    /// that may ignore it, even if that family's router is the default.
    pub(super) fn select_router_for_adapter(
        &self,
        model: &str,
        adapter: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        if adapter.is_some() && has_sglang_grpc(&self.worker_registry, model) {
            self.select_lora_router(model)
        } else {
            self.select_router_for_request(Some(model))
        }
    }

    /// A composite SGLang selector must stay on the tokenized SGLang path:
    /// an HTTP or external family may ignore the normalized LoRA extension.
    pub(super) fn select_lora_router(&self, model: &str) -> Option<Arc<dyn RouterTrait>> {
        if !self.enable_igw {
            let default = self.default_router.load();
            let default = default.as_ref()?;
            return if [
                router_ids::GRPC_REGULAR,
                router_ids::GRPC_PD,
                router_ids::GRPC_EPD,
            ]
            .contains(&default.id)
            {
                default.router.clone()
            } else {
                None
            };
        }
        let snapshot = self.worker_registry.get_routing_snapshot(model);
        let size = |pool| {
            snapshot
                .pool(pool)
                .iter()
                .filter(|worker| sglang_grpc(worker.as_ref()))
                .count()
        };
        let weights = self.family_weights(size);
        self.pick_router_by_weights(&weights).or_else(|| {
            self.incomplete_disaggregated_router(
                size(RoutingPool::GrpcEncode),
                size(RoutingPool::GrpcPrefill) + size(RoutingPool::GrpcDecode),
                0,
            )
        })
    }
}

#[cfg(test)]
mod tests;
