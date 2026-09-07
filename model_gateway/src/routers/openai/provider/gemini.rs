use serde_json::Value;

use super::{Provider, ProviderError};
use crate::{
    routers::common::sglang_fields::strip_sglang_fields,
    worker::{Endpoint, ProviderType},
};

pub struct GeminiProvider;

impl Provider for GeminiProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Gemini
    }

    fn transform_request(
        &self,
        payload: &mut Value,
        endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        strip_sglang_fields(payload);

        if endpoint == Endpoint::Chat {
            if let Some(obj) = payload.as_object_mut() {
                if obj.get("logprobs").and_then(|v| v.as_bool()) == Some(false) {
                    obj.remove("logprobs");
                }
            }
        }
        Ok(())
    }
}
