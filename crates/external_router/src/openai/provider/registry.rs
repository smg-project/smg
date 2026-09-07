use std::{collections::HashMap, sync::Arc};

use openai_protocol::worker::ProviderType;

use super::{
    AnthropicProvider, GeminiProvider, OpenAIProvider, Provider, SGLangProvider, XAIProvider,
};

pub struct ProviderRegistry {
    providers: HashMap<ProviderType, Arc<dyn Provider>>,
    default_provider: Arc<dyn Provider>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    pub fn new() -> Self {
        let mut providers = HashMap::new();

        providers.insert(
            ProviderType::OpenAI,
            Arc::new(OpenAIProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::XAI,
            Arc::new(XAIProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::Gemini,
            Arc::new(GeminiProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::Anthropic,
            Arc::new(AnthropicProvider) as Arc<dyn Provider>,
        );

        Self {
            providers,
            default_provider: Arc::new(SGLangProvider),
        }
    }

    pub fn get_arc(&self, provider_type: &ProviderType) -> Arc<dyn Provider> {
        self.providers
            .get(provider_type)
            .cloned()
            .unwrap_or_else(|| Arc::clone(&self.default_provider))
    }

    pub fn default_provider_arc(&self) -> Arc<dyn Provider> {
        Arc::clone(&self.default_provider)
    }
}
