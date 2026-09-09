//! Gemini Interactions API router implementation
//!
//! This module implements the Gemini Interactions API using a state machine pattern.
//! The state machine drives request processing through explicit states and steps,
//! supporting both streaming and non-streaming flows with MCP tool interception.
//!
//! # Architecture
//!
//! - **State machine**: A single `execute()` driver dispatches steps based on `RequestState`.
//! - **Two-level context**: `SharedComponents` (per-router) and `RequestContext` (per-request).
//! - **Tool loop**: MCP tool calls cause state transitions back to the execution state,
//!   forming an explicit loop in the state machine rather than a nested `loop {}`.

mod context;
mod driver;
mod router;
mod state;
mod steps;

pub use router::GeminiRouter;
