//! What a worker states of the media it takes, read from its labels, its
//! runtime and its wire. The reading has one home: a host's selection and a
//! host's registration both read a worker's media facts here, so the two
//! cannot part.

use super::{HOST_PIPELINE_PROCESSOR, MM_PROCESSOR_LABEL, SUPPORTS_VISION_LABEL};
use crate::identity::basic::{ConnectionMode, RuntimeType, Worker};

/// Whether the worker's engine runs without a vision encoder, so no pixel
/// payload may reach it: its vision label reads `"false"`
/// ([`SUPPORTS_VISION_LABEL`]). A worker without the label reads as one
/// that takes images.
pub fn language_model_only(worker: &dyn Worker) -> bool {
    worker
        .metadata()
        .spec
        .labels
        .get(SUPPORTS_VISION_LABEL)
        .is_some_and(|value| value == "false")
}

/// Whether `worker` fetches and processes media references itself: a gRPC
/// worker whose media processor label ([`MM_PROCESSOR_LABEL`]) names a
/// processor its runtime serves references with.
pub fn accepts_media_refs(worker: &dyn Worker) -> bool {
    let spec = &worker.metadata().spec;
    *worker.connection_mode() == ConnectionMode::Grpc
        && spec
            .labels
            .get(MM_PROCESSOR_LABEL)
            .is_some_and(|value| match spec.runtime_type {
                RuntimeType::Vllm => !value.is_empty(),
                RuntimeType::Sglang => names_host_pipeline(value),
                _ => false,
            })
}

/// Whether `worker` processes media references with its host's own
/// pipeline ([`HOST_PIPELINE_PROCESSOR`]) rather than its engine's.
pub fn runs_host_pipeline(worker: &dyn Worker) -> bool {
    accepts_media_refs(worker)
        && worker
            .metadata()
            .spec
            .labels
            .get(MM_PROCESSOR_LABEL)
            .is_some_and(|value| names_host_pipeline(value))
}

fn names_host_pipeline(processor: &str) -> bool {
    processor
        .trim()
        .eq_ignore_ascii_case(HOST_PIPELINE_PROCESSOR)
}
