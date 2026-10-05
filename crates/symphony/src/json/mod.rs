//! JSON as models write it: values that are still arriving.
//!
//! A tool call arrives as a JSON object that grows with every chunk. [`outline`](fn@outline) finds the call's
//! name and the byte span of its arguments in that growing text, so an argument stream can emit the
//! model's own bytes as they arrive; [`partial`] parses a JSON prefix into the value it determines so
//! far, for the places that need the value rather than its bytes. [`assembler`] turns one call's
//! growing object into the events of that call, on top of the outline.

pub mod assembler;
pub mod outline;
pub mod partial;

pub use assembler::Assembler;
pub use outline::{outline, Outline, Span};
pub use partial::{is_complete, PartialJson, PartialJsonError};
