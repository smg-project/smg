//! Pipeline stages for regular (non-harmony) model processing
//!
//! This module defines stages specific to regular tokenizer-based models.

pub(crate) mod chat;
pub(crate) mod classify;
pub(crate) mod completion;
pub(crate) mod embedding;
pub(crate) mod generate;
pub(crate) mod messages;
pub(crate) mod preparation;
pub(crate) mod request_building;
pub(crate) mod response_processing;
pub(crate) mod transcription;

// Re-export chat+generate dispatcher stages
pub(crate) use preparation::ChatGeneratePreparationStage;
pub(crate) use request_building::ChatGenerateRequestBuildingStage;
pub(crate) use response_processing::ChatGenerateResponseProcessingStage;
pub(crate) use transcription::{
    TranscriptionPreparationStage, TranscriptionRequestBuildingStage,
    TranscriptionResponseProcessingStage,
};
