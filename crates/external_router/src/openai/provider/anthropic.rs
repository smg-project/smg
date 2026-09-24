use openai_protocol::worker::ProviderType;

use super::Provider;

pub struct AnthropicProvider;

impl Provider for AnthropicProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }
}
