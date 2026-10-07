//! Tool calls written as tagged parameters.
//!
//! Several model families write a call as a function tag followed by one tag per argument, the
//! value as plain text between the parameter's opening and closing tag: Qwen 3.5 and later and
//! Qwen3-Coder (`<function=name>`, `<parameter=key>`), DeepSeek's DSML, MiniMax, GLM and Hy, each
//! with its own spellings. What they share is that the arguments are built by the parser, not
//! passed through: the model writes no JSON object, so the parser writes one, and it has to decide
//! what type each value's text is.
//!
//! [`value`] is that decision; [`assembler`] turns one Qwen call's tags into its events,
//! streaming a declared string as it arrives and writing every other value at its close; [`dsml`]
//! does the same for one DeepSeek DSML invoke, whose `string` attribute types each value.

pub mod assembler;
pub mod dsml;
pub mod value;

pub use assembler::Assembler;
pub use value::{json, Declared, Kind};
