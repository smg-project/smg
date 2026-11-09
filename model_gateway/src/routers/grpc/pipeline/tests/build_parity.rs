//! The pipeline builder against its hand-transcribed descriptor per endpoint and mode, so a wrong
//! mode mapping fails here.

use super::*;
use crate::routers::grpc::mode::Mode;

impl RequestPipeline {
    /// Mode-bearing construction descriptor for the parity test. Stage
    /// ordering itself is structural (see `PipelineStages`), so only the
    /// per-mode args and optional stages need freezing.
    fn signature(&self) -> String {
        format!(
            "{}|encode={}|{}",
            self.stages.worker_selection.signature(),
            self.stages.encode.is_some(),
            self.stages.request_building.signature(),
        )
    }
}

/// Hand-transcribed expected construction descriptor + metrics backend
/// label per endpoint/mode. Do not regenerate from `build()` output, or
/// this stops guarding anything: it exists to catch a wrong
/// `build`/`mode.rs` mapping (a flipped `inject_pd_metadata`, wrong
/// `plan_kind`, or swapped `WorkerSelectionMode`). Stage ordering is
/// structural in `PipelineStages` and needs no golden.
fn golden(endpoint: Endpoint, mode: Mode) -> (String, &'static str) {
    const REGULAR: &str = metrics_labels::BACKEND_REGULAR;
    const PD: &str = metrics_labels::BACKEND_PD;
    let (selection, inject, plan, backend) = match mode {
        Mode::Regular => ("Regular", "false", "Single", REGULAR),
        Mode::PrefillDecode => ("PrefillDecode", "true", "PrefillDecode", PD),
        Mode::EncodePrefillDecode => ("EncodePrefillDecode", "false", "EncodePrefillDecode", PD),
    };
    let encode = matches!(mode, Mode::EncodePrefillDecode);
    let build = match endpoint {
        Endpoint::Chat => format!(
            "ChatGenerateRequestBuildingStage(ChatRequestBuildingStage(inject_pd_metadata={inject}, {plan}), GenerateRequestBuildingStage(inject_pd_metadata={inject}, {plan}))"
        ),
        Endpoint::Messages => {
            format!("MessageRequestBuildingStage(inject_pd_metadata={inject}, {plan})")
        }
        Endpoint::Completion => {
            format!("CompletionRequestBuildingStage(inject_pd_metadata={inject}, {plan})")
        }
        Endpoint::Harmony => {
            format!("HarmonyRequestBuildingStage(inject_pd_metadata={inject}, {plan})")
        }
        Endpoint::Embeddings | Endpoint::Classify => {
            "EmbeddingRequestBuildingStage".to_string()
        }
        Endpoint::Transcription => "TranscriptionRequestBuildingStage".to_string(),
        Endpoint::Decisions => "DecisionsRequestBuildingStage".to_string(),
    };
    // Harmony never carries the encode stage.
    let encode = encode && !matches!(endpoint, Endpoint::Harmony);
    (
        format!("WorkerSelectionStage({selection})|encode={encode}|{build}"),
        backend,
    )
}

fn assert_parity(endpoint: Endpoint, mode: Mode, deps: &PipelineDeps) {
    let (expected_sig, expected_backend) = golden(endpoint, mode);
    let built = RequestPipeline::build(endpoint, mode, deps)
        .unwrap_or_else(|| panic!("build({endpoint:?}, {mode:?}) should be valid"));
    assert_eq!(
        built.signature(),
        expected_sig,
        "construction parity for {endpoint:?}/{mode:?}"
    );
    assert_eq!(
        built.backend_type, expected_backend,
        "backend_type parity for {endpoint:?}/{mode:?}"
    );
    assert_eq!(built.mode, mode, "mode parity for {endpoint:?}/{mode:?}");
}

#[test]
fn build_matches_frozen_goldens() {
    let deps = PipelineDeps::test_default();

    for endpoint in [Endpoint::Chat, Endpoint::Messages, Endpoint::Completion] {
        for mode in [
            Mode::Regular,
            Mode::PrefillDecode,
            Mode::EncodePrefillDecode,
        ] {
            assert_parity(endpoint, mode, &deps);
        }
    }

    assert!(
        RequestPipeline::build(Endpoint::Harmony, Mode::EncodePrefillDecode, &deps).is_none(),
        "Harmony EPD must be invalid"
    );
    assert_parity(Endpoint::Harmony, Mode::Regular, &deps);
    assert_parity(Endpoint::Harmony, Mode::PrefillDecode, &deps);

    for endpoint in [
        Endpoint::Embeddings,
        Endpoint::Classify,
        Endpoint::Transcription,
        Endpoint::Decisions,
    ] {
        assert!(
            RequestPipeline::build(endpoint, Mode::PrefillDecode, &deps).is_none(),
            "{endpoint:?} PD must be invalid"
        );
        assert!(
            RequestPipeline::build(endpoint, Mode::EncodePrefillDecode, &deps).is_none(),
            "{endpoint:?} EPD must be invalid"
        );
        assert_parity(endpoint, Mode::Regular, &deps);
    }
}
