use openai_protocol::model_type::Endpoint;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Unsupported endpoint: {0:?}")]
    UnsupportedEndpoint(Endpoint),

    #[error("Transform error: {0}")]
    TransformError(String),
}
