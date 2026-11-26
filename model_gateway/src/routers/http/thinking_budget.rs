use axum::response::Response;
use serde_json::{value::to_raw_value, Map, Value};
use smg_worker::{
    identity::basic::{RuntimeType, Worker},
    labels::SUPPORTS_THINKING_TOKEN_BUDGET_LABEL,
};

use super::request_body::{RawBody, RequestBodyError};
use crate::routers::error;

/// Both HTTP routers keep the canonical field until the worker for this
/// attempt is known. Only SGLang needs an edit to the serialized request.
pub(super) fn needs_nested_budget(worker: &dyn Worker) -> Result<bool, String> {
    let spec = &worker.metadata().spec;
    let advertised = spec
        .labels
        .get(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL)
        .is_some_and(|v| v == "true");
    match spec.runtime_type {
        RuntimeType::Trtllm => Ok(false),
        RuntimeType::Vllm if advertised => Ok(false),
        RuntimeType::Sglang if advertised => Ok(true),
        runtime => Err(format!(
            "the selected {runtime} worker does not advertise support for thinking_token_budget"
        )),
    }
}

fn nest_budget(body: &mut RawBody<'_>, budget: u32) -> Result<(), String> {
    let mut params: Map<String, Value> = match body.get("custom_params") {
        Some(raw) if raw.get() != "null" => serde_json::from_str(raw.get())
            .map_err(|_| "custom_params must be an object to set a thinking budget on SGLang")?,
        _ => Map::new(),
    };
    params.insert("thinking_budget".into(), budget.into());
    body.insert(
        "custom_params",
        to_raw_value(&params).map_err(|e| e.to_string())?,
    );
    body.remove("thinking_token_budget");
    Ok(())
}

pub(super) fn prepare(
    bytes: Vec<u8>,
    worker: &dyn Worker,
    budget: u32,
    raw_len: Option<usize>,
) -> Result<Vec<u8>, RequestBodyError> {
    if !needs_nested_budget(worker).map_err(RequestBodyError::ThinkingBudget)? {
        return Ok(bytes);
    }
    let mut body = RawBody::parse(&bytes).map_err(RequestBodyError::Serialize)?;
    nest_budget(&mut body, budget).map_err(RequestBodyError::ThinkingBudget)?;
    body.to_bytes(raw_len).map_err(RequestBodyError::Serialize)
}

pub(super) fn prepare_pair<'a>(
    budget: Option<u32>,
    prefill: (&mut RawBody<'a>, &dyn Worker),
    decode: (&mut RawBody<'a>, &dyn Worker),
) -> Result<(), Box<Response>> {
    let Some(budget) = budget else { return Ok(()) };
    let refuse = |e| Box::new(error::bad_request("unsupported_thinking_token_budget", e));
    for (body, worker) in [prefill, decode] {
        if needs_nested_budget(worker).map_err(refuse)? {
            nest_budget(body, budget).map_err(refuse)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use openai_protocol::chat::ChatCompletionRequest;
    use serde_json::json;

    use super::*;
    use crate::routers::{grpc::backend_client::BasicWorkerBuilder, http::chat_for_upstream};

    #[test]
    fn thinking_token_budget_reaches_each_native_http_format() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Trtllm, RuntimeType::Vllm] {
            let worker = BasicWorkerBuilder::new("http://worker")
                .runtime_type(runtime)
                .labels([(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL.into(), "true".into())].into())
                .build();
            for fields in [
                json!({"thinking_token_budget":0,"custom_params":{"other":42}}),
                json!({"custom_params":{"thinking_budget":0,"other":42}}),
            ] {
                let mut input = json!({"model":"m","messages":[{"role":"user","content":"hi"}]});
                input
                    .as_object_mut()
                    .unwrap()
                    .extend(fields.as_object().unwrap().clone());
                let req: ChatCompletionRequest = serde_json::from_value(input).unwrap();
                let req = chat_for_upstream(req).unwrap();
                let bytes = serde_json::to_vec(&req).unwrap();
                let bytes = prepare(
                    bytes,
                    &worker,
                    req.resolved_thinking_token_budget().unwrap().unwrap(),
                    None,
                )
                .unwrap();
                let wire: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(wire["custom_params"]["other"], 42);
                if runtime == RuntimeType::Sglang {
                    assert!(wire.get("thinking_token_budget").is_none());
                    assert_eq!(wire["custom_params"]["thinking_budget"], 0);
                } else {
                    assert_eq!(wire["thinking_token_budget"], 0);
                    assert!(wire["custom_params"].get("thinking_budget").is_none());
                }
            }
        }
    }

    #[test]
    fn thinking_token_budget_requires_a_capable_worker() {
        for runtime in [
            RuntimeType::Sglang,
            RuntimeType::Vllm,
            RuntimeType::Mlx,
            RuntimeType::Generic,
        ] {
            let worker = BasicWorkerBuilder::new("http://worker")
                .runtime_type(runtime)
                .build();
            assert!(prepare(b"{}".to_vec(), &worker, 0, None).is_err());
        }
    }
}
