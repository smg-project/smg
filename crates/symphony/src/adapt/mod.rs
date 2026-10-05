//! Protocol adapters: events to response deltas, with all response policy in one place.
//!
//! Each API gets one module. [`chat`] renders Chat Completions and [`responses`] the Responses API;
//! Messages follows. An adapter is a pure function over [`Event`](crate::Event)s: what a client sees
//! for each event can be read in one place and tested without a parser, and the driver that owns the
//! request keeps only what belongs to the response as a whole (ids, model, timestamps, usage).

pub mod chat;
pub mod responses;
