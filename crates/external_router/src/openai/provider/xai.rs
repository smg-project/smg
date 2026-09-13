use openai_protocol::{model_type::Endpoint, worker::ProviderType};
use serde_json::Value;

use super::{Provider, ProviderError};
use crate::sglang_fields::strip_sglang_fields;

pub struct XAIProvider;

impl Provider for XAIProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::XAI
    }

    fn transform_request(
        &self,
        payload: &mut Value,
        endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        strip_sglang_fields(payload);

        if endpoint == Endpoint::Responses {
            if let Some(obj) = payload.as_object_mut() {
                Self::transform_responses_input(obj);
            }
        }
        Ok(())
    }
}

impl XAIProvider {
    /// Rewrite Responses API input items to the shape xAI accepts today.
    ///
    /// The only content-part normalization xAI requires is `output_text ->
    /// input_text` on historical messages replayed from `previous_response_id`
    /// chains. Every other content variant (`input_text`, `input_image`,
    /// `input_file`, `refusal`) is passed through verbatim so that file and
    /// image inputs reach the upstream provider unchanged.
    fn transform_responses_input(obj: &mut serde_json::Map<String, Value>) {
        let Some(input_arr) = obj.get_mut("input").and_then(Value::as_array_mut) else {
            return;
        };

        for item in input_arr.iter_mut().filter_map(Value::as_object_mut) {
            item.remove("id");
            item.remove("status");

            let Some(content_arr) = item.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };

            for content in content_arr.iter_mut().filter_map(Value::as_object_mut) {
                if content.get("type").and_then(Value::as_str) == Some("output_text") {
                    content.insert("type".to_string(), Value::String("input_text".to_string()));
                }
            }
        }
    }
}
