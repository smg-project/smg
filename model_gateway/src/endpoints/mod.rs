//! Endpoint handlers that do not route inference traffic.
//!
//! These handlers serve requests directly from the application context:
//! tokenizer management and tokenize/detokenize, tool-call and reasoning
//! parsing, conversation storage, and stored-response retrieval. None of
//! them select a worker or touch a router, so they live outside `routers`.

pub mod conversations;
pub mod parse;
pub mod responses;
pub mod tokenize;
