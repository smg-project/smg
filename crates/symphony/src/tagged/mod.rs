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
//! streaming a declared string as it arrives and writing every other value at its close;
//! [`keyed`] does the same for one call written as its name and `<arg_key>`/`<arg_value>` pairs
//! (GLM, Hy4, Ling, IQuest), typed by the request's tools again; [`attributed`] for one invoke
//! whose parameter tags name their key in an attribute (MiniMax M2, dots3, MiniCPM5, typed by the
//! request's tools; DeepSeek's DSML, typed by the tag's `string` attribute; Kimi K3's XTML, typed
//! by the tag's `type` attribute); and [`xml`] for one MiniMax M3 invoke, whose arguments are an
//! XML tree, typed by the request's tools at every depth. [`object`] is the arguments object the
//! keyed and attributed assemblers write: the carried bytes, the separators and the close, and the
//! three ways a member is written.

pub mod assembler;
pub mod attributed;
pub mod keyed;
mod object;
pub mod value;
pub mod xml;

pub use assembler::{Assembler, Placement, Spelling};
pub use value::{json, Declared, Kind};
