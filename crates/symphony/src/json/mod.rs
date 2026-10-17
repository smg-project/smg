//! JSON as models write it: values that are still arriving.
//!
//! A tool call arrives as a JSON object that grows with every chunk. `outline`'s scanner finds the call's
//! name and the byte span of its arguments in that growing text, so an argument stream can emit the
//! model's own bytes as they arrive; `partial` parses a JSON prefix into the value it determines so
//! far, for the places that need the value rather than its bytes; `prefix` judges, byte by byte
//! as they arrive, whether the argument bytes are still a prefix of a value. `assembler` turns
//! one call's growing object into the events of that call, on top of the outline and the prefix;
//! `named` reads a call whose name stands before a marker, with the call's object or the
//! arguments object alone (`named_arguments`) after it, both object readers accounting for the
//! argument bytes they send, hold or report through one judge (`fragments`); `list` reads a
//! list of call objects; `arguments` a call's arguments alone, written as one value, the call
//! named elsewhere; `forced` runs it for the call the request forces, whose arguments are the
//! whole output.

pub(crate) mod arguments;
pub(crate) mod assembler;
pub(crate) mod forced;
mod fragments;
mod header;
pub(crate) mod list;
pub(crate) mod named;
mod named_arguments;
pub(crate) mod outline;
pub(crate) mod partial;
pub(crate) mod prefix;

pub(crate) use arguments::Arguments;
pub(crate) use assembler::Assembler;
pub use partial::{PartialJson, PartialJsonError};
pub(crate) use prefix::Prefix;
