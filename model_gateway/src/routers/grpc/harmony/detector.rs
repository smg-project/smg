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
    /// This method looks up workers from the registry by model name and checks
    /// if any of them are Harmony workers based on their metadata (architectures,
    /// hf_model_type).
    ///
    /// Falls back to string-based detection if no workers are registered for
    /// the model (e.g., during startup before workers are discovered).
    pub fn is_harmony_model_in_registry(registry: &WorkerRegistry, model_name: &str) -> bool {
        // Get workers for this model
        let workers = registry.get_by_model(model_name);

        if workers.is_empty() {
            // No workers found - fall back to string-based detection
            return Self::is_harmony_model(model_name);
        }

        // Check if any worker is a Harmony worker
        workers
            .iter()
            .any(|worker| Self::is_harmony_worker(worker.as_ref()))
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
    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, WorkerType};

    fn builder() -> BasicWorkerBuilder {
        BasicWorkerBuilder::new("grpc://127.0.0.1:1")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
    }

    /// A worker whose metadata holds no card yet, served as `model_id`.
    fn undiscovered(model_id: &str) -> impl Worker {
        builder().label("model_id", model_id).build()
    }

    /// A card found gpt_oss by its model type after discovery, on a worker
    /// whose metadata held none: the lazily discovered list is what the
    /// check reads.
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
}
