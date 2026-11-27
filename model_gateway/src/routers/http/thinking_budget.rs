use axum::response::Response;
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
pub(super) fn translate(
    body: &mut RawBody<'_>,
    format: BudgetWire,
    budget: i64,
) -> Result<(), String> {
    match format {
        BudgetWire::Sglang if body.contains(TOP) => {
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
        BudgetWire::Sglang | BudgetWire::Passthrough => {}
        BudgetWire::Canonical => {
            remove_nested_budget(body)?;
            body.insert(TOP, to_raw_value(&budget).map_err(|e| e.to_string())?);
        }
    }
    Ok(())
}

pub(super) fn prepare_regular(
    bytes: Vec<u8>,
    budget: Option<i64>,
    worker: &dyn Worker,
) -> Result<Vec<u8>, RequestBodyError> {
    let Some(budget) = budget else {
        return Ok(bytes);
    };
    let format = wire(worker.metadata().spec.runtime_type);
    if format == BudgetWire::Passthrough {
        return Ok(bytes);
    }
    let mut body = RawBody::parse(&bytes).map_err(RequestBodyError::Serialize)?;
    translate(&mut body, format, budget).map_err(RequestBodyError::ThinkingBudget)?;
    if body.mutated() {
        body.to_bytes(Some(bytes.len()))
            .map_err(RequestBodyError::Serialize)
    } else {
        Ok(bytes)
    }
}

pub(super) fn prepare_pair<'a>(
    budget: Option<i64>,
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
    fn both_budget_spellings_reach_each_http_runtime_in_its_native_field() {
        let top = json!({"thinking_token_budget":0,"custom_params":{"other":1}});
        let nested = json!({"custom_params":{"thinking_budget":0,"other":1}});
        let prefill = BasicWorkerBuilder::new("http://prefill")
            .runtime_type(RuntimeType::Sglang)
            .build();
        for input in [&top, &nested] {
            let sent = serde_json::to_vec(input).unwrap();
            for (runtime, expected) in [
                (RuntimeType::Sglang, &nested),
                (RuntimeType::Vllm, &top),
                (RuntimeType::Trtllm, &top),
                (RuntimeType::Generic, input),
            ] {
                let worker = BasicWorkerBuilder::new("http://worker")
                    .runtime_type(runtime)
                    .build();
                let bytes = prepare_regular(sent.clone(), Some(0), &worker).unwrap();
                assert_eq!(
                    &serde_json::from_slice::<Value>(&bytes).unwrap(),
                    expected,
                    "{runtime:?}"
                );
                let mut prefill_body = RawBody::parse(&sent).unwrap();
                let mut decode_body = prefill_body.clone();
                prepare_pair(
                    Some(0),
                    (&mut prefill_body, &prefill),
                    (&mut decode_body, &worker),
                )
                .unwrap();
                assert_eq!(serde_json::to_value(&prefill_body).unwrap(), nested);
                assert_eq!(&serde_json::to_value(&decode_body).unwrap(), expected);
            }
        }
    }
}
