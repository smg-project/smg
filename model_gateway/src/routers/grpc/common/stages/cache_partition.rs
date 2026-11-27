//! The request fields that partition an engine's prefix cache, on the gRPC
//! wire. This module reads the fields and stamps them on the built requests.
//! It also says which workers apply them.
//!
//! A salt can be a secret. No log line or error text here holds a value.
//! They name the field.

use std::time::Duration;

use axum::response::Response;
use openai_protocol::common::{CachePartition, GenerationRequest};
use serde_json::{Map, Value};
use smg_worker::{identity::basic::Worker, labels::CACHE_PARTITION_FIELDS_LABEL};
use tracing::warn;

use crate::{
    observability::log_throttle::KeyedLogThrottle,
    routers::{
        common::placement,
        error,
        grpc::{
            context::{ExecutionPlan, RequestType},
            proto_wrapper::ProtoGenerateRequest,
        },
    },
};

/// The code of the 400 for a field that no worker of the model applies.
pub(crate) const NOT_SUPPORTED_CODE: &str = "cache_partition_not_supported";

const CACHE_SALT: &str = "cache_salt";
const EXTRA_KEY: &str = "extra_key";

/// The lines of that refusal, a model at a time under the refusal lines'
/// own allowance.
static REFUSAL_LOG: KeyedLogThrottle = KeyedLogThrottle::new(
    placement::REFUSAL_LINES_PER_SECOND,
    Duration::from_secs(1),
    placement::REFUSAL_LOG_MODELS,
);

/// The request's partition as the gRPC wire forwards it. The salt and the
/// extra key come from the same reader as on the HTTP path. The LoRA adapter
/// is the one that the request builders forward. An empty value is unset.
/// Call this after [`refuse_unforwarded`]: the reader reads a non-string value
/// as unset.
pub(crate) fn request_partition(request_type: &RequestType) -> CachePartition<'_> {
    let body = match request_type {
        RequestType::Chat(request) => request.cache_partition(),
        RequestType::Generate(request) => request.cache_partition(),
        RequestType::Completion(request) => request.cache_partition(),
        RequestType::Messages(request) => request.cache_partition(),
        RequestType::Responses(_)
        | RequestType::Embedding(_)
        | RequestType::Classify(_)
        | RequestType::Decisions(_)
        | RequestType::Transcription { .. } => CachePartition::default(),
    };
    CachePartition {
        cache_salt: body.cache_salt.filter(|value| !value.is_empty()),
        extra_key: body.extra_key.filter(|value| !value.is_empty()),
        lora_path: request_type.lora_adapter(),
    }
}

/// Refuse a field value that the gRPC wire cannot carry. One case is a value
/// that is not a string. The other is a value on a request kind that does not
/// forward the field. Without this check, the engine gets no value and the
/// client gets no error.
pub(crate) fn refuse_unforwarded(request_type: &RequestType) -> Result<(), Response> {
    let (body, forwarded): (&Map<String, Value>, bool) = match request_type {
        RequestType::Chat(request) => (&request.other, true),
        RequestType::Generate(request) => (&request.other, true),
        RequestType::Completion(request) => (&request.other, true),
        RequestType::Messages(request) => (&request.other, true),
        RequestType::Embedding(request) => (&request.other, false),
        RequestType::Classify(request) => (&request.other, false),
        RequestType::Decisions(request) => (&request.other, false),
        RequestType::Responses(_) | RequestType::Transcription { .. } => return Ok(()),
    };
    for field in [CACHE_SALT, EXTRA_KEY] {
        match body.get(field) {
            None | Some(Value::Null) => {}
            Some(Value::String(value)) if value.is_empty() || forwarded => {}
            Some(Value::String(_)) => {
                return Err(error::bad_request(
                    NOT_SUPPORTED_CODE,
                    format!(
                        "the gateway does not forward {field} over gRPC on {request_type} requests"
                    ),
                ));
            }
            Some(_) => {
                return Err(error::bad_request(
                    "invalid_request_parameters",
                    format!("{field} must be a string"),
                ));
            }
        }
    }
    Ok(())
}

/// Give each generate request of the plan the request's salt and extra key.
/// The request builders set the LoRA adapter. This function does not change it.
pub(crate) fn stamp(plan: &mut ExecutionPlan, partition: CachePartition<'_>) {
    if !CachePartitionDemand::of(&partition).any() {
        return;
    }
    for request in plan.generate_requests_mut() {
        request.set_cache_partition(partition);
    }
}

/// The partition fields a request carries. Each worker that runs the request
/// must give all of them to its engine. An engine without a field shares its
/// prefix cache across the values of that field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CachePartitionDemand {
    cache_salt: bool,
    extra_key: bool,
}

impl CachePartitionDemand {
    pub(crate) fn of(partition: &CachePartition<'_>) -> Self {
        let set = |field: Option<&str>| field.is_some_and(|value| !value.is_empty());
        Self {
            cache_salt: set(partition.cache_salt),
            extra_key: set(partition.extra_key),
        }
    }

    /// Whether the request carries a field.
    pub(crate) fn any(self) -> bool {
        self.cache_salt || self.extra_key
    }

    /// Whether `worker` gives every field the request carries to its engine.
    /// Its `cache_partition_fields` label must list each field. The gateway's
    /// wire for its runtime must also carry each field.
    pub(crate) fn met_by(self, worker: &dyn Worker) -> bool {
        if !self.any() {
            return true;
        }
        let listed = worker
            .metadata()
            .spec
            .labels
            .get(CACHE_PARTITION_FIELDS_LABEL)
            .map_or("", String::as_str);
        let carried =
            ProtoGenerateRequest::cache_partition_fields(worker.metadata().spec.runtime_type);
        let lists = |field: &str| {
            carried.contains(&field) && listed.split(',').any(|name| name.trim() == field)
        };
        (!self.cache_salt || lists(CACHE_SALT)) && (!self.extra_key || lists(EXTRA_KEY))
    }

    fn fields(self) -> &'static str {
        match (self.cache_salt, self.extra_key) {
            (true, true) => "cache_salt and extra_key",
            (true, false) => CACHE_SALT,
            (false, true) => EXTRA_KEY,
            (false, false) => "no field",
        }
    }

    /// The 400 for a request whose fields none of the model's `workers`
    /// give to their engine. A 400 is terminal for the retry layer.
    pub(crate) fn refusal(self, model_id: &str, workers: &str) -> Response {
        let fields = self.fields();
        if let Some(admitted) = REFUSAL_LOG.admit(model_id) {
            warn!(
                function = "WorkerSelectionStage::execute",
                model_id = %model_id,
                workers,
                fields,
                code = NOT_SUPPORTED_CODE,
                suppressed = admitted.suppressed,
                "Refused a request whose cache partition fields its workers do not give to their engines"
            );
        }
        error::bad_request(
            NOT_SUPPORTED_CODE,
            format!(
                "model '{model_id}' cannot apply {fields} over gRPC. Its {workers} do not \
                 give {fields} to their engines. An engine that drops a field shares its prefix \
                 cache across the values of that field."
            ),
        )
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/stages/cache_partition/tests.rs"]
mod tests;
