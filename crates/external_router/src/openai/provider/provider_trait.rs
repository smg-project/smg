use openai_protocol::{model_type::Endpoint, worker::ProviderType};
use reqwest::RequestBuilder;
use serde_json::Value;

use super::ProviderError;
use crate::sglang_fields::strip_sglang_fields;

/// Default `transform_request` strips SGLang fields.
pub trait Provider: Send + Sync {
    fn provider_type(&self) -> ProviderType;

    fn transform_request(
        &self,
        payload: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        strip_sglang_fields(payload);
        Ok(())
    }

    fn transform_response(
        &self,
        _response: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Ok(())
    }

    fn apply_headers(&self, builder: RequestBuilder) -> RequestBuilder {
        builder
    }
}
