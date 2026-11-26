use axum::response::Response;
use openai_protocol::common::ThinkingTokenBudget;
use serde_json::{value::to_raw_value, Map, Value};
use smg_worker::identity::basic::{RuntimeType, Worker};

use super::request_body::{RawBody, RequestBodyError};
use crate::routers::error;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BudgetWire {
    Sglang,
    Canonical,
    Passthrough,
}

pub(super) fn wire(runtime: RuntimeType) -> BudgetWire {
    match runtime {
        RuntimeType::Sglang => BudgetWire::Sglang,
        RuntimeType::Vllm | RuntimeType::Trtllm => BudgetWire::Canonical,
        _ => BudgetWire::Passthrough,
    }
}

const TOP: &str = "thinking_token_budget";
const NESTED: &str = "thinking_budget";

fn remove_nested_budget(body: &mut RawBody<'_>) -> Result<(), String> {
    if let Some(mut params) = body
        .get("custom_params")
        .and_then(|raw| serde_json::from_str::<Map<String, Value>>(raw.get()).ok())
        .filter(|params| params.contains_key(NESTED))
    {
        params.remove(NESTED);
        if params.is_empty() {
            body.remove("custom_params");
        } else {
            body.insert(
                "custom_params",
                to_raw_value(&params).map_err(|e| e.to_string())?,
            );
        }
    }
    Ok(())
}

// Only the budget occupies two positions in the client JSON. Keep other
// custom parameters on the request, even if the upstream rejects them.
fn translate(
    body: &mut RawBody<'_>,
    format: BudgetWire,
    budget: ThinkingTokenBudget,
) -> Result<(), String> {
    match (format, budget) {
        (BudgetWire::Passthrough, _) => {}
        (_, ThinkingTokenBudget::NullAlias) => {
            remove_nested_budget(body)?;
            body.remove(TOP);
        }
        (BudgetWire::Sglang, ThinkingTokenBudget::Tokens(budget)) if body.contains(TOP) => {
            let mut params: Map<String, Value> = match body.get("custom_params") {
                Some(raw) if raw.get() != "null" => {
                    serde_json::from_str(raw.get()).map_err(|_| {
                        "custom_params must be an object to set a thinking budget on SGLang"
                    })?
                }
                _ => Map::new(),
            };
            params.insert(NESTED.into(), budget.into());
            body.insert(
                "custom_params",
                to_raw_value(&params).map_err(|e| e.to_string())?,
            );
            body.remove(TOP);
        }
        (BudgetWire::Sglang, ThinkingTokenBudget::Tokens(_)) => {}
        (BudgetWire::Canonical, ThinkingTokenBudget::Tokens(budget)) => {
            remove_nested_budget(body)?;
            body.insert(TOP, to_raw_value(&budget).map_err(|e| e.to_string())?);
        }
    }
    Ok(())
}

pub(super) fn prepare(
    bytes: Vec<u8>,
    worker: &dyn Worker,
    budget: ThinkingTokenBudget,
    raw_len: Option<usize>,
) -> Result<Vec<u8>, RequestBodyError> {
    let format = wire(worker.metadata().spec.runtime_type);
    if format == BudgetWire::Passthrough {
        return Ok(bytes);
    }
    let mut body = RawBody::parse(&bytes).map_err(RequestBodyError::Serialize)?;
    translate(&mut body, format, budget).map_err(RequestBodyError::ThinkingBudget)?;
    if body.mutated() {
        body.to_bytes(raw_len).map_err(RequestBodyError::Serialize)
    } else {
        Ok(bytes)
    }
}

pub(super) fn prepare_pair<'a>(
    budget: Option<ThinkingTokenBudget>,
    prefill: (&mut RawBody<'a>, &dyn Worker),
    decode: (&mut RawBody<'a>, &dyn Worker),
) -> Result<(), Box<Response>> {
    let Some(budget) = budget else { return Ok(()) };
    for (body, worker) in [prefill, decode] {
        translate(body, wire(worker.metadata().spec.runtime_type), budget)
            .map_err(|e| Box::new(error::bad_request("invalid_thinking_token_budget", e)))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    #[test]
    fn thinking_token_budget_uses_the_selected_http_wire() {
        for budget in [-1, 0, 64] {
            let canonical = json!({"thinking_token_budget":budget, "custom_params":{"sibling":42}});
            let nested = json!({"custom_params":{"thinking_budget":budget,"sibling":42}});
            let both = json!({"thinking_token_budget":budget,"custom_params":{"thinking_budget":7,"sibling":42}});
            for input in [&canonical, &nested, &both] {
                for (runtime, expected) in [
                    (RuntimeType::Sglang, &nested),
                    (RuntimeType::Vllm, &canonical),
                    (RuntimeType::Trtllm, &canonical),
                    (RuntimeType::Generic, input),
                    (RuntimeType::Unspecified, input),
                ] {
                    let worker = BasicWorkerBuilder::new("http://worker")
                        .runtime_type(runtime)
                        .build();
                    let bytes = prepare(
                        serde_json::to_vec(input).unwrap(),
                        &worker,
                        ThinkingTokenBudget::Tokens(budget),
                        None,
                    )
                    .unwrap();
                    assert_eq!(&serde_json::from_slice::<Value>(&bytes).unwrap(), expected);
                }
            }
        }
    }

    #[tokio::test]
    async fn null_thinking_token_budget_alias_is_omitted_for_known_http_workers() {
        use axum::{
            body::Body,
            extract::{FromRequest, Request},
            http::header::CONTENT_TYPE,
        };
        use openai_protocol::{
            chat::ChatCompletionRequest, common::GenerationRequest, validated::ValidatedJson,
        };

        let worker = |runtime| {
            BasicWorkerBuilder::new("http://worker")
                .runtime_type(runtime)
                .build()
        };
        let trt = worker(RuntimeType::Trtllm);
        let sglang = worker(RuntimeType::Sglang);
        let generic = worker(RuntimeType::Generic);
        for (fields, expected_params) in [
            (json!({"custom_params": {"thinking_budget": null}}), None),
            (
                json!({"thinking_token_budget": null, "custom_params": {"thinking_budget": null, "keep": 42}}),
                Some(json!({"keep": 42})),
            ),
        ] {
            let mut input = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
            input
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let request = Request::builder()
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(input.to_string()))
                .unwrap();
            let ValidatedJson(chat) =
                ValidatedJson::<ChatCompletionRequest>::from_request(request, &())
                    .await
                    .unwrap();
            let budget = chat.thinking_token_budget().unwrap();
            assert_eq!(budget, ThinkingTokenBudget::NullAlias);
            assert_eq!(chat.resolved_thinking_token_budget().unwrap(), None);
            let bytes = serde_json::to_vec(&chat).unwrap();

            let regular = prepare(bytes.clone(), &trt, budget, None).unwrap();
            let mut prefill = RawBody::parse(&bytes).unwrap();
            let mut decode = prefill.clone();
            prepare_pair(
                Some(budget),
                (&mut prefill, &sglang),
                (&mut decode, &generic),
            )
            .unwrap();
            for native in [regular, prefill.to_bytes(None).unwrap()] {
                let body: Value = serde_json::from_slice(&native).unwrap();
                assert!(body.get(TOP).is_none());
                assert_eq!(body.get("custom_params"), expected_params.as_ref());
            }
            let passthrough: Value =
                serde_json::from_slice(&decode.to_bytes(None).unwrap()).unwrap();
            assert_eq!(passthrough["custom_params"], input["custom_params"]);
        }
    }

    #[test]
    fn thinking_token_budget_removes_only_the_alias_for_strict_http_servers() {
        let trt = BasicWorkerBuilder::new("http://worker")
            .runtime_type(RuntimeType::Trtllm)
            .build();
        let bytes = prepare(
            br#"{"custom_params":{"thinking_budget":0}}"#.to_vec(),
            &trt,
            ThinkingTokenBudget::Tokens(0),
            None,
        )
        .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            json!({"thinking_token_budget":0})
        );
    }
}
