//! Tool calls written as tagged parameters.
//!
//! Several model families write a call as a function tag followed by one tag per argument, the
//! value as plain text between the parameter's opening and closing tag: Qwen 3.5 and later and
//! Qwen3-Coder (`<function=name>`, `<parameter=key>`), DeepSeek's DSML, MiniMax, GLM and Hy, each
//! with its own spellings. What they share is that the arguments are built by the parser, not
//! passed through: the model writes no JSON object, so the parser writes one, and it has to decide
//! what type each value's text is.
//!
//! `value` is that decision; `assembler` turns one Qwen call's tags into its events,
//! streaming a declared string as it arrives and writing every other value at its close;
//! `keyed` does the same for one call written as its name and `<arg_key>`/`<arg_value>` pairs
//! (GLM, Hy4, Ling, IQuest), typed by the request's tools again; `attributed` for one invoke
//! whose parameter tags name their key in an attribute (MiniMax M2, dots3, MiniCPM5, typed by the
//! request's tools; DeepSeek's DSML, typed by the tag's `string` attribute; Kimi K3's XTML, typed
//! by the tag's `type` attribute); and `xml` for one MiniMax M3 invoke, whose arguments are an
//! XML tree, typed by the request's tools at every depth. `object` is the arguments object the
//! keyed and attributed assemblers write: the carried bytes, the separators and the close, and the
//! three ways a member is written.

pub(crate) mod assembler;
pub(crate) mod attributed;
pub(crate) mod keyed;
mod object;
pub(crate) mod value;
pub(crate) mod xml;

pub(crate) use assembler::Assembler;
pub use assembler::{Placement, Spelling};

use crate::event::{DropReason, Event, Events, MalformedReason, Text};

/// The end of a started call that was cut, for the four assemblers of this module and the pythonic
/// one: by the end of the stream, whatever stage the cut found; by the call's closing marker (the
/// region's, or the next call's opener) inside a streamed string; or, for the pythonic one, by the
/// region's marker at any stage of a call (its arguments, a key, a value), which
/// `pythonic::Assembler::close` reports through here as well. Each writes the arguments object
/// itself and closes it only at the call's own end, so the fragments sent are no whole JSON value:
/// a string left open, a value held for its close, or no member yet. The fragments stand, since a
/// fragment passed on cannot be taken back, nothing is made up for what the model never finished
/// (no closing quote, no closing brace), and one `Malformed` event before the call's end says that
/// the arguments are no whole value, as the JSON assembler says it: it carries the bytes still held
/// (the start of a tag, a value's text, the template's newline; a pythonic key or separator cut
/// short), or none when the cut fell between events or the syntax refused what it held first (the
/// keyed one, [`refuse`]), so that a client that reads the arguments as JSON is told rather than
/// left to find out. The reason is `UnterminatedRegion`, which the engine keeps for a region the
/// end of the stream cut and renames for one a marker closed; `terminal` is the marker's bytes when
/// they are the call's own end, and nothing when they are the region's or the stream ended.
pub(crate) fn end_cut_call(index: u32, held: String, terminal: &str, out: &mut Events) {
    out.push(Event::Malformed {
        text: Text::uncounted(held),
        why: MalformedReason::UnterminatedRegion,
    });
    out.push(Event::ToolCallEnd {
        index,
        source: Text::uncounted(terminal.to_owned()),
    });
}

/// Bytes of a call block the syntax could not read, refused: the failure is counted by a
/// `Malformed` event of no bytes with `why`, which stands before the bytes so that a reader's
/// offset points at them, and the bytes leave as `Dropped { MalformedCall }`, never as content.
/// For the keyed syntax, whose block is never prose: its markers and its four tags are the
/// format's own tokens, and what stands between them where the syntax has no place for text (a
/// value whose key never opened, a name that is prose or Python, a tag written twice) is the
/// model's failed call, which the families' own parsers drop whole; a client shown it saw the
/// format's tags inside the answer. A call that started keeps the pairs it read. Nothing is
/// pushed for no bytes; the engine renames `UnterminatedRegion` here as anywhere.
pub(crate) fn refuse(text: String, why: MalformedReason, out: &mut Events) {
    if text.is_empty() {
        return;
    }
    out.push(Event::Malformed {
        text: Text::uncounted(String::new()),
        why,
    });
    drop_refused(text, out);
}

/// The bytes of a refused call block alone, when the event that counts the failure comes on its
/// own: before a cut call's end ([`end_cut_call`]), which says the arguments are no whole value.
pub(crate) fn drop_refused(text: String, out: &mut Events) {
    if text.is_empty() {
        return;
    }
    out.push(Event::Dropped {
        text: Text::uncounted(text),
        why: DropReason::MalformedCall,
    });
}
