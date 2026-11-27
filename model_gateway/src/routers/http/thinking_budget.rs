use axum::response::Response;
use serde_json::{value::to_raw_value, Map, Value};
use smg_worker::identity::basic::{RuntimeType, Worker};

use super::request_body::RawBody;
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
        RuntimeType::Vllm => BudgetWire::Canonical,
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
    fn thinking_token_budget_uses_the_selected_http_wire() {
        for budget in [-1, 0, 64] {
            let canonical = json!({"thinking_token_budget":budget, "custom_params":{"sibling":42}});
            let nested = json!({"custom_params":{"thinking_budget":budget,"sibling":42}});
            let both = json!({"thinking_token_budget":budget,"custom_params":{"thinking_budget":7,"sibling":42}});
            let aligned = json!({"thinking_token_budget":budget,"custom_params":{"thinking_budget":budget,"sibling":42}});
            for (input, passthrough) in [
                (&canonical, &canonical),
                (&nested, &nested),
                (&both, &aligned),
            ] {
                let mut chat = json!({"model":"m","messages":[]});
                chat.as_object_mut()
                    .unwrap()
                    .extend(input.as_object().unwrap().clone());
                let chat = super::super::chat_for_upstream(serde_json::from_value(chat).unwrap());
                for (runtime, expected) in [
                    (RuntimeType::Sglang, &nested),
                    (RuntimeType::Vllm, &canonical),
                    (RuntimeType::Generic, passthrough),
                    (RuntimeType::Unspecified, passthrough),
                ] {
                    let worker = BasicWorkerBuilder::new("http://worker")
                        .runtime_type(runtime)
                        .build();
                    let bytes = super::super::request_body::serialize_request_body_with_edits(
                        &chat,
                        None,
                        &worker,
                        None,
                        true,
                        Some(budget),
                    )
                    .unwrap();
                    let sent: Value = serde_json::from_slice(&bytes).unwrap();
                    for key in [TOP, "custom_params"] {
                        assert_eq!(sent.get(key), expected.get(key), "{runtime:?}: {input}");
                    }
                }
            }
        }
    }
}
