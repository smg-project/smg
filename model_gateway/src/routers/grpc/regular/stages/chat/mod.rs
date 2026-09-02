//! Chat endpoint pipeline stages
//!
//! These stages handle chat-specific preprocessing, request building, and response processing.
//! They work with any model type by using injected model adapters.

mod preparation;
mod request_building;
mod response_processing;

pub(crate) use preparation::{prepare_chat_like, ChatPreparationStage};
pub(crate) use request_building::{build_chat_backed_plan, ChatRequestBuildingStage};
pub(crate) use response_processing::ChatResponseProcessingStage;
