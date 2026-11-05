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

/// The one event of a call, or a string value in it, that the stream's stop closed: the model's
/// own end of output found the call open (the closing marker never came) or a streamed string
/// open (its closing tag never came), and the assembler closes what the model left open, since
/// the model wrote nothing after its bytes. The event carries no bytes and stands before the
/// fragment that closes what was open, the quote or the brace, so that a reader counts the
/// shape where the fragments it accounts for begin ([`MalformedReason::OpenAtTheStop`]). The
/// four assemblers of this module push it from their `stop`; a call cut by the token limit, an
/// abort or any finish but `Stop` never comes here ([`end_cut_call`]).
pub(crate) fn open_at_the_stop(out: &mut Events) {
    left_open(MalformedReason::OpenAtTheStop, out);
}

/// The one event of a string value that one of a call's two own ends closed: the stream's stop
/// (`why` = [`MalformedReason::OpenAtTheStop`], [`open_at_the_stop`]) or the call's closing
/// marker (`why` = [`MalformedReason::OpenAtTheMarker`]: the model ended the call, its own
/// closing marker, the next call's opener or the block's close came, with the string's closing
/// tag never written). Either way the model wrote nothing more of the value, so the assembler
/// closes the string; the event carries no bytes and stands before the closing quote. The four
/// assemblers of this module push it from their `close_open_string`; a value held whole for its
/// close keeps the ending's own path, reported or refused, nothing made up for it.
pub(crate) fn left_open(why: MalformedReason, out: &mut Events) {
    out.push(Event::Malformed {
        text: Text::uncounted(String::new()),
        why,
    });
}

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
/// short), or none when the cut fell between events, the syntax refused what it held first (the
/// keyed one, [`refuse`]) or wrote it as the call's own fragment, dropping the wrapping before the
/// end and a cut tag after it (the attributed one at the end of the stream), so that a client that
/// reads the arguments as JSON is told rather than left to find out. The reason is
/// `UnterminatedRegion`, which the engine keeps for a region the end of the stream cut and renames
/// for one a marker closed; `terminal` is the marker's bytes when they are the call's own end, and
/// nothing when they are the region's or the stream ended.
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
/// format's tags inside the answer. For the attributed syntax inside an invoke the same (a word
/// between tags, a tag out of its place or cut short, a tail the tag has no place for, an
/// invoke that named no call). A call that started keeps the pairs it read. Nothing is pushed
/// for no bytes; the engine renames `UnterminatedRegion` here as anywhere.
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

/// The text of a value held whole for its close: the carried bytes from `start` to `end`, or to
/// their end. A value is held by its position in the carried bytes, which holds only while
/// nothing takes them from under it ([`Object::carried`](object::Object::carried)); a position
/// past them reads as no text, never as a panic, since a reader never panics on bytes.
pub(super) fn held_text(carried: &str, start: usize, end: Option<usize>) -> &str {
    let end = end.unwrap_or(carried.len()).min(carried.len());
    debug_assert!(start <= end, "a value held past the carried bytes");
    carried.get(start..end).unwrap_or_default()
}

/// No fragment is empty and no `Malformed` or `Dropped` text is empty, but for the one event
/// of no bytes that counts a refused run and stands right before its `Dropped {
/// MalformedCall }`: any other empty report would claim a byte where none was.
#[cfg(test)]
pub(crate) fn nothing_empty(events: &[Event]) {
    for (at, event) in events.iter().enumerate() {
        match event {
            Event::ToolCallArguments { json, .. } => assert!(!json.is_empty(), "{event:?}"),
            Event::Malformed { text, .. } if text.text.is_empty() => assert!(
                matches!(
                    events.get(at + 1),
                    Some(Event::Dropped { text, why: DropReason::MalformedCall })
                        if !text.text.is_empty()
                ),
                "{event:?}: an empty report with no refused bytes after it"
            ),
            Event::Malformed { text, .. } | Event::Dropped { text, .. } => {
                assert!(!text.text.is_empty(), "{event:?}");
            }
            _ => {}
        }
    }
}

/// The bytes dropped as the template's wrapper, in order: the whitespace around the syntax's
/// tags, and for the attributed syntax a tag a marker cut short.
#[cfg(test)]
pub(crate) fn dropped(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Dropped {
                text,
                why: DropReason::Wrapper,
            } => Some(text.text.to_string()),
            _ => None,
        })
        .collect()
}

/// The bytes refused: the text of the `Dropped { MalformedCall }` events, which no adapter
/// shows.
#[cfg(test)]
pub(crate) fn refused(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Dropped {
                text,
                why: DropReason::MalformedCall,
            } => Some(text.text.to_string()),
            _ => None,
        })
        .collect()
}

/// The content a chat client reads: every content and malformed text, in order, as the chat
/// adapter folds it.
#[cfg(test)]
pub(crate) fn shown(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Content(text) | Event::Malformed { text, .. } => Some(&*text.text),
            _ => None,
        })
        .collect()
}
