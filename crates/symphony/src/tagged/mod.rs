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
//! XML tree, typed by the request's tools at every depth. `object` is the arguments object the
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

use crate::event::{Event, Events, MalformedReason, Text};

/// The end of a started call the stream cut, for the four assemblers of this module. Each writes
/// the arguments object itself and closes it only at the call's own end, so when the stream ends
/// first the fragments sent are no whole JSON value, whatever stage the cut found: a string left
/// open, a value held for its close, or no member yet. The fragments stand, since a fragment
/// passed on cannot be taken back, nothing is made up for what the model never finished, and one
/// `Malformed` event before the call's end says that the arguments are no whole value, as the
/// JSON assembler says it: it carries the bytes still held (the start of a tag, a value's text,
/// the template's newline), or none when the cut fell between events, so that a client that
/// reads the arguments as JSON is told rather than left to find out. The engine keeps
/// `UnterminatedRegion` for a region the end of the stream cut.
pub(crate) fn end_cut_call(index: u32, held: String, out: &mut Events) {
    out.push(Event::Malformed {
        text: Text::uncounted(held),
        why: MalformedReason::UnterminatedRegion,
    });
    out.push(Event::ToolCallEnd {
        index,
        source: Text::default(),
    });
}
