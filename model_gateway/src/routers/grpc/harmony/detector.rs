//! Harmony model detection

use crate::worker::{Worker, WorkerRegistry};

/// Harmony model detector
///
/// Detects if a model name indicates support for Harmony encoding/parsing.
pub(crate) struct HarmonyDetector;

impl HarmonyDetector {
    /// Check if a worker is a Harmony/GPT-OSS model.
    ///
    /// Detection priority:
    /// 1. Check if any model card has architectures containing "GptOssForCausalLM"
    /// 2. Check if any model card has hf_model_type equal to "gpt_oss"
    /// 3. Check if model_id contains "gpt-oss" substring (case-insensitive)
    pub fn is_harmony_worker(worker: &dyn Worker) -> bool {
        // The cards are read in place: this runs on every chat request, for
        // every worker of the model, and `models()` would clone each card.
        let harmony_card = worker.any_model(&mut |model_card| {
            is_harmony_card(
                &model_card.architectures,
                model_card.hf_model_type.as_deref(),
                &model_card.id,
            )
        });
        if harmony_card {
            return true;
        }

        // Fallback: check worker's model_id directly
        Self::is_harmony_model(worker.model_id())
    }

    /// Check if a model name contains "gpt-oss" (case-insensitive).
    pub fn is_harmony_model(model_name: &str) -> bool {
        // Case-insensitive substring search without heap allocation
        // More efficient than to_lowercase() which allocates a new String
        model_name
            .as_bytes()
            .windows(7) // "gpt-oss".len()
            .any(|window| window.eq_ignore_ascii_case(b"gpt-oss"))
    }

    /// Check if any worker for the given model is a Harmony/GPT-OSS worker.
    ///
    /// The model's workers come from the registry's membership snapshot for
    /// the model (alias-aware), and the snapshot keeps the answer: every
    /// worker's cards are read by [`Self::is_harmony_worker`] once per
    /// membership generation, not on every chat and Responses request.
    ///
    /// Falls back to string-based detection if no workers are registered for
    /// the model (e.g., during startup before workers are discovered).
    pub fn is_harmony_model_in_registry(registry: &WorkerRegistry, model_name: &str) -> bool {
        match registry.model_routing_snapshot(model_name) {
            Some(snapshot) if !snapshot.is_empty() => snapshot.harmony(|workers| {
                workers
                    .iter()
                    .any(|worker| Self::is_harmony_worker(worker.as_ref()))
            }),
            // No workers found - fall back to string-based detection
            _ => Self::is_harmony_model(model_name),
        }
    }
}

/// Whether a model card describes a Harmony/GPT-OSS model: an architecture
/// `GptOssForCausalLM`, the Hugging Face model type `gpt_oss`, or an id that
/// contains `gpt-oss` (any case). `pub` (re-exported as
/// `routers::grpc::is_harmony_card`) so that the Python binding reads a
/// checkpoint's `config.json` by the rule serving reads a worker's cards by.
pub fn is_harmony_card(architectures: &[String], hf_model_type: Option<&str>, id: &str) -> bool {
    architectures.iter().any(|arch| arch == "GptOssForCausalLM")
        || hf_model_type == Some("gpt_oss")
        || HarmonyDetector::is_harmony_model(id)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, WorkerType};

    fn builder() -> BasicWorkerBuilder {
        builder_at("grpc://127.0.0.1:1")
    }

    fn builder_at(url: &str) -> BasicWorkerBuilder {
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
    }

    fn gpt_oss_card(id: &str) -> ModelCard {
        let mut card = ModelCard::new(id);
        card.hf_model_type = Some("gpt_oss".to_string());
        card
    }

    /// A worker whose metadata holds no card yet, served as `model_id`.
    fn undiscovered(model_id: &str) -> impl Worker {
        builder().label("model_id", model_id).build()
    }

    /// A card found gpt_oss by its model type after discovery, on a worker
    /// whose metadata held none: the lazily discovered list is what
    /// `is_harmony_worker` reads (the registry check reads the answer its
    /// membership snapshot kept).
    #[test]
    fn a_discovered_card_with_the_gpt_oss_model_type_is_harmony() {
        let worker = undiscovered("served-model");
        assert!(
            !HarmonyDetector::is_harmony_worker(&worker),
            "no card, a plain id"
        );
        let mut discovered = ModelCard::new("served-model");
        discovered.hf_model_type = Some("gpt_oss".to_string());
        worker.set_models(vec![discovered]);
        assert!(HarmonyDetector::is_harmony_worker(&worker));
    }

    #[test]
    fn a_card_with_the_gpt_oss_architecture_is_harmony_whatever_its_id() {
        let mut card = ModelCard::new("plain-7b");
        card.architectures = vec!["GptOssForCausalLM".to_string()];
        let worker = builder().model(card).build();
        assert!(HarmonyDetector::is_harmony_worker(&worker));
    }

    #[test]
    fn a_worker_whose_id_says_gpt_oss_is_harmony_through_the_fallback() {
        let worker = undiscovered("openai/GPT-OSS-20b");
        assert!(HarmonyDetector::is_harmony_worker(&worker));
    }

    #[test]
    fn a_plain_card_and_id_are_not_harmony() {
        let worker = builder().model(ModelCard::new("plain-7b")).build();
        assert!(!HarmonyDetector::is_harmony_worker(&worker));
        assert!(!HarmonyDetector::is_harmony_worker(&undiscovered(
            "plain-7b"
        )));
    }

    /// The registry's answer follows the model's membership: a plain
    /// worker's model is not Harmony; a Harmony worker registered for the
    /// same model makes it so; its removal takes the answer back. Each
    /// membership change publishes a new snapshot, so the kept answer of the
    /// old generation never outlives it.
    #[test]
    fn the_registry_answer_follows_the_models_membership() {
        let registry = WorkerRegistry::new();
        let plain: Arc<dyn Worker> = Arc::new(
            builder_at("grpc://127.0.0.1:1")
                .model(ModelCard::new("shared-7b"))
                .build(),
        );
        registry
            .register(plain)
            .expect("the plain worker registers");
        assert!(!HarmonyDetector::is_harmony_model_in_registry(
            &registry,
            "shared-7b"
        ));
        assert!(
            !HarmonyDetector::is_harmony_model_in_registry(&registry, "shared-7b"),
            "the second read of a generation reads the kept answer"
        );

        let harmony: Arc<dyn Worker> = Arc::new(
            builder_at("grpc://127.0.0.1:2")
                .model(gpt_oss_card("shared-7b"))
                .build(),
        );
        let harmony_id = registry
            .register(harmony)
            .expect("the Harmony worker registers");
        assert!(HarmonyDetector::is_harmony_model_in_registry(
            &registry,
            "shared-7b"
        ));

        registry.remove(&harmony_id);
        assert!(!HarmonyDetector::is_harmony_model_in_registry(
            &registry,
            "shared-7b"
        ));
    }

    /// A worker rebuilt with other cards (a property update re-registers the
    /// worker in place) decides the next generation's answer.
    #[test]
    fn a_rebuilt_workers_cards_decide_the_next_generation() {
        let registry = WorkerRegistry::new();
        let plain: Arc<dyn Worker> = Arc::new(
            builder_at("grpc://127.0.0.1:1")
                .model(ModelCard::new("served-model"))
                .build(),
        );
        let worker_id = registry.register(plain).expect("the worker registers");
        assert!(!HarmonyDetector::is_harmony_model_in_registry(
            &registry,
            "served-model"
        ));

        let rebuilt: Arc<dyn Worker> = Arc::new(
            builder_at("grpc://127.0.0.1:1")
                .model(gpt_oss_card("served-model"))
                .build(),
        );
        assert!(registry.replace(&worker_id, rebuilt));
        assert!(HarmonyDetector::is_harmony_model_in_registry(
            &registry,
            "served-model"
        ));
    }

    /// A model no worker serves is read by its name alone, as before.
    #[test]
    fn a_model_without_workers_is_read_by_its_name() {
        let registry = WorkerRegistry::new();
        assert!(HarmonyDetector::is_harmony_model_in_registry(
            &registry,
            "openai/gpt-oss-20b"
        ));
        assert!(!HarmonyDetector::is_harmony_model_in_registry(
            &registry, "plain-7b"
        ));
    }
}
