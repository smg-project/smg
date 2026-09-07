use openai_protocol::worker::ProviderType;

use super::Provider;

pub struct OpenAIProvider;

impl Provider for OpenAIProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }
}
