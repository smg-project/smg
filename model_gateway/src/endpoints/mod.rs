//! Endpoint handlers that do not route inference traffic.
//!
//! These handlers serve requests directly from the application context:
//! tokenizer management and tokenize/detokenize, tool-call and reasoning
//! parsing, conversation storage, stored-response retrieval, and the model
//! inventory. None of them select a worker or touch a router, so they live
//! outside `routers`. `messages` holds the Messages route's typed header
//! extractor, which the route's handlers in `server` take.

pub mod conversations;
pub mod messages;
pub mod models;
pub mod parse;
pub mod responses;
pub mod tokenize;
