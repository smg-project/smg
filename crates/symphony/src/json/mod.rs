//! JSON as models write it: values that are still arriving.
//!
//! A tool call arrives as a JSON object that grows with every chunk. [`outline`](fn@outline) finds the call's
//! name and the byte span of its arguments in that growing text, so an argument stream can emit the
//! model's own bytes as they arrive; [`partial`] parses a JSON prefix into the value it determines so
//! far, for the places that need the value rather than its bytes; [`prefix`] judges, byte by byte
//! as they arrive, whether the argument bytes are still a prefix of a value. [`assembler`] turns
//! one call's growing object into the events of that call, on top of the outline and the prefix;
//! [`named_arguments`] does the same for a call whose name stands before a marker and whose object
//! is the arguments alone; [`arguments`] for a call's arguments alone, written as one value, the
//! call named elsewhere; [`forced`] runs it for the call the request forces, whose arguments are
//! the whole output.

pub mod arguments;
pub mod assembler;
pub mod forced;
pub mod list;
pub mod named;
pub mod named_arguments;
pub mod outline;
pub mod partial;
pub mod prefix;

pub use arguments::Arguments;
pub use assembler::Assembler;
pub use outline::{outline, Outline, Span};
pub use partial::{is_complete, PartialJson, PartialJsonError};
pub use prefix::Prefix;
