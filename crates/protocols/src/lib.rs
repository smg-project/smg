// Protocol definitions and validation for various LLM APIs
// This module provides a structured approach to handling different API protocols

// The worker's wire types, the model card and the default model identifier
// live in the worker crate; the paths they had here are kept.
#[doc(inline)]
pub use smg_worker::{model_card, model_type, worker, UNKNOWN_MODEL_ID};

pub mod builders;
pub mod chat;
pub mod classify;
pub mod common;
pub mod completion;
pub mod decisions;
pub mod embedding;
pub mod event_types;
pub mod ext;
pub mod generate;
pub mod interactions;
mod kind_read;
pub mod messages;
mod messages_validation;
pub mod models;
pub mod multipart;
pub mod parser;
pub mod profile;
pub mod realtime_conversation;
pub mod realtime_events;
pub mod realtime_response;
pub mod realtime_session;
pub mod rerank;
pub mod responses;
pub mod rl;
pub mod sampling_params;
mod strict_value;
pub mod systemone;
pub mod tokenize;
pub mod transcription;
pub mod validated;
