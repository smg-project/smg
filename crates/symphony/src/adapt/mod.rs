//! Protocol adapters: events to response deltas, with all response policy in one place.
//!
//! Each API gets one module: [`chat`] renders Chat Completions, [`responses`] the Responses API and
//! [`messages`] the Messages API. An adapter is a pure function over [`Event`](crate::Event)s: what a
//! client sees for each event can be read in one place and tested without a parser, and the driver
//! that owns the request keeps only what belongs to the response as a whole (ids, model, timestamps,
//! usage).

pub mod chat;
pub mod messages;
pub mod responses;
