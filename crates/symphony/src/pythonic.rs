//! Pythonic calls: the events of the calls a model writes as Python, `name(key=value, ...)`.
//!
//! Olmo 3 writes a turn's calls between `<function_calls>` and `</function_calls>`, one per line:
//!
//! ```text
//! <function_calls>get_weather(city="Paris", days=3)
//! get_stock_price()</function_calls>
//! ```
//!
//! LFM2.5 writes them as a Python list between `<|tool_call_start|>` and `<|tool_call_end|>`:
//!
//! ```text
//! <|tool_call_start|>[get_weather(city='Paris', days=3), get_stock_price()]<|tool_call_end|>
//! ```
//!
//! Llama 4 writes the list with no markers at all, right after the turn's content when there is
//! one, and every value between double quotes:
//!
//! ```text
//! Let me check.[get_weather(city="Paris", days="3"), get_stock_price()]
//! ```
//!
//! The engine's table owns the markers; this assembler reads what lies between them: one call or
//! several, bare or in a list, and gives each its own index. Where no marker frames the list
//! ([`Assembler::bare`]), it reads the whole turn: the list where the turn starts with one, after
//! whitespace alone, as it arrives; content otherwise, and after content a list only once it is
//! whole and well formed; the template's wrapping after the list.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    ids::CallIds,
    tagged::{
        end_cut_call,
        value::{json, python_literal, Kind},
    },
    Declared,
};

/// How a template frames the calls of one turn, which the grammar derived from the table writes
/// and the assembler, which reads either, does not need: Olmo 3 writes them one per line between
/// its markers, LFM2.5 and Llama 4 as a Python list, `[` and `]` around calls joined by `, `.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// One call per line: `get_weather(city="Paris")\nget_stock_price()`.
    OnePerLine,
    /// A Python list: `[get_weather(city='Paris'), get_stock_price()]`.
    List,
}

/// How a template writes a value: LFM2.5's template writes Python (`True`, `None`, a string in
/// either quote); Olmo 3's writes every value through `tojson`, so JSON (`true`, `null`, a string
/// in double quotes); Llama 4's writes every value between double quotes, whatever its type. The
/// grammar derived from the table writes the dialect's spelling, and the assembler reads every
/// spelling; the quotes of the last one carry no type, so it reads a quoted value by the type
/// the request's tool declares for the parameter ([`Declared`]), as a tagged value reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Values {
    /// `True`, `False`, `None`, `'Paris'` or `"Paris"`.
    Python,
    /// `true`, `false`, `null`, `"Paris"`.
    Json,
    /// `"Paris"`, `"3"`, `"True"`, `"['a', 'b']"`: every value as `"%s"` of it, the quotes
    /// carrying no type and the text between them the value's characters as they are, no escape
    /// among them. A value between double quotes reads as written, by the kind the request's tool
    /// declares for its parameter, by its text where none is declared; a value the model writes
    /// bare, a single-quoted string included, reads as Python's.
    Quoted,
}

/// How a template spells its pythonic calls: how it frames them and how it writes a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dialect {
    pub framing: Framing,
    pub values: Values,
}

impl Dialect {
    /// Olmo 3: one call per line, each value as `tojson` writes it.
    pub const OLMO3: Self = Self {
        framing: Framing::OnePerLine,
        values: Values::Json,
    };
    /// LFM2.5: a Python list of calls with Python values.
    pub const LFM2_5: Self = Self {
        framing: Framing::List,
        values: Values::Python,
    };
    /// Llama 4: a Python list of calls, every value between double quotes.
    pub const LLAMA4: Self = Self {
        framing: Framing::List,
        values: Values::Quoted,
    };
}

/// The most bytes a list held after content may grow to before it is given back as content. A
/// hold whose first name no tool declares is given back at its `(` already, so this is the
/// backstop for a declared name followed by a stray quote, and it sits far above any real call's
/// arguments: a call that writes a whole file stays a call.
const HOLD_CAP_BYTES: usize = 256 * 1024;

/// The events of the pythonic calls in one region, each call with its own index from `first` and
/// its id from the engine's [`CallIds`].
#[derive(Clone, Debug)]
pub struct Assembler {
    /// The index of the next call to start.
    next_index: u32,
    /// Where the calls' ids count from.
    ids: CallIds,
    /// The current call's index, once it started.
    index: u32,
    /// The current call's name, for the kind its tool declares for a parameter.
    name: String,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    carried: String,
    stage: Stage,
    /// Members written to the current call's arguments object so far.
    written: u32,
    /// How the template writes a value.
    values: Values,
    /// Whether the list stands bare in the turn, no marker before or after it: the turn is the
    /// list, with the template's wrapping after it, or content.
    bare: bool,
    /// Under the bare list: whether the turn has read anything but whitespace before the list's
    /// `[`; after that a list is held until it is whole.
    begun: bool,
}

#[derive(Clone, Debug)]
enum Stage {
    /// Under the bare list: the turn's content. `carried` holds the content not yet given back:
    /// whitespace alone until the turn has begun, which may still be a list's wrapping.
    Prose,
    /// Under the bare list: a `[` arrived, and `carried` holds it with the whitespace and the name
    /// after it, until a `(` says it opens a list of calls or any other character gives it back as
    /// content.
    Opening { named: bool, spaced: bool },
    /// Under the bare list, after content: a list opened, and `carried` holds it until its `]`
    /// closes it, when it reads as the calls it holds if it is well formed and as content if not.
    Holding(ListExtent),
    /// Before a call's name: the list's brackets, separators and whitespace pass as wrapping.
    Between,
    /// Inside a name: `carried` holds it so far; `spaced` once whitespace followed it, after which
    /// only `(` can continue the call.
    Name { spaced: bool },
    /// After `(` or an argument's `,`: a key or `)` comes next.
    Arguments,
    /// Inside a key: the key is `carried[start..]`, less whitespace before the `=`; `spaced` once
    /// whitespace followed it.
    Key { start: usize, spaced: bool },
    /// Inside a value: its text is `carried[start..]`, scanned for its extent.
    Value {
        key: String,
        start: usize,
        scan: Extent,
    },
    /// Under the bare list: after the list's `]`. `carried` holds the current run, whitespace or
    /// not; whitespace is the template's wrapping, anything else malformed.
    After,
}

/// Where a value ends: at a `,` or `)` outside every string and bracket.
#[derive(Clone, Copy, Debug, Default)]
struct Extent {
    depth: u32,
    quote: Option<char>,
    escaped: bool,
}

impl Extent {
    /// Reads one character and says whether it ends the value (it is then not the value's).
    fn ends_at(&mut self, c: char) -> bool {
        if let Some(quote) = self.quote {
            if self.escaped {
                self.escaped = false;
            } else if c == '\\' {
                self.escaped = true;
            } else if c == quote {
                self.quote = None;
            }
            return false;
        }
        match c {
            '\'' | '"' => self.quote = Some(c),
            '(' | '[' | '{' => self.depth += 1,
            ')' | ']' | '}' if self.depth > 0 => self.depth -= 1,
            ',' | ')' if self.depth == 0 => return true,
            _ => {}
        }
        false
    }
}

/// Where a list held after content ends: at the `]` that closes the `[` it opened with, outside
/// every string and nested bracket.
#[derive(Clone, Copy, Debug)]
struct ListExtent {
    depth: u32,
    quote: Option<char>,
    escaped: bool,
}

impl ListExtent {
    /// The extent after the list's `[`.
    fn opened() -> Self {
        Self {
            depth: 1,
            quote: None,
            escaped: false,
        }
    }

    /// Reads one character and says what it leaves: the list still open, closed by its `]`, or
    /// closed by another bracket, which no list of calls writes.
    fn read(&mut self, c: char) -> ListEnd {
        if let Some(quote) = self.quote {
            if self.escaped {
                self.escaped = false;
            } else if c == '\\' {
                self.escaped = true;
            } else if c == quote {
                self.quote = None;
            }
            return ListEnd::Open;
        }
        match c {
            '\'' | '"' => self.quote = Some(c),
            '(' | '[' | '{' => self.depth += 1,
            ')' | ']' | '}' => {
                self.depth -= 1;
                if self.depth == 0 {
                    return if c == ']' {
                        ListEnd::Closed
                    } else {
                        ListEnd::Mismatched
                    };
                }
            }
            _ => {}
        }
        ListEnd::Open
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListEnd {
    Open,
    Closed,
    Mismatched,
}

/// Whether `c` can start a name.
fn starts_a_name(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Whether `c` can continue a name: Python's identifier characters, and the `.` and `-` a tool's
/// name may hold.
fn continues_a_name(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '.' || c == '-'
}

impl Assembler {
    /// An assembler for the calls of one region the table's markers frame, the first taking
    /// index `first`, their ids counting from `ids`.
    pub fn new(first: u32, ids: CallIds, dialect: Dialect) -> Self {
        Self {
            next_index: first,
            ids,
            index: first,
            name: String::new(),
            carried: String::new(),
            stage: Stage::Between,
            written: 0,
            values: dialect.values,
            bare: false,
            begun: false,
        }
    }

    /// An assembler for the list bare in the turn, no marker before or after it (Llama 4), the
    /// first call taking index `first`. A turn that starts, after whitespace alone, with a `[` that
    /// a name and its `(` follow is the list, read as it arrives, with the whitespace as its
    /// wrapping. A turn that starts with anything else is content; a `[name(` after that content,
    /// the shape the template renders for a message that holds both, is held until the `]` that
    /// closes it and read as its calls only if the whole is a well-formed list of calls with
    /// keyword arguments to tools the request declares, since code holds `[name(` of its own
    /// (`[f(a+h) - f(a)]/h`, `[Authorize(Roles = "Manager")]`); anything else, a list the stream's
    /// end cut included, is content. After the list's `]` whitespace is
    /// the template's wrapping and anything else malformed, as it is after xLAM's bare list. A `[`
    /// that no call follows (`[Note]`, `[1, 2]`, a markdown link) is content too.
    pub fn bare(first: u32, ids: CallIds, dialect: Dialect) -> Self {
        Self {
            stage: Stage::Prose,
            bare: true,
            ..Self::new(first, ids, dialect)
        }
    }

    /// Append the next bytes of the region and push the events they complete. Every byte is the
    /// region's until the engine says otherwise, so this takes them all. `declared` types a
    /// quoted value where the dialect's quotes carry no type.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        for c in bytes.chars() {
            self.take(c, declared, out);
        }
        // Content and the text after a bare list are given back as they arrive, run by run; a
        // `[` under consideration and a list held after content stay held, and so does the
        // whitespace before the turn's first byte, which a list would make its wrapping.
        match self.stage {
            Stage::Prose if self.begun => self.release_content(out),
            Stage::After => self.release_after(out),
            _ => {}
        }
    }

    /// The region's end: its closing marker, or the next opener. A call still open is one the
    /// model never finished, since every template writes the call's `)` before the marker: it
    /// ends as it does at the stream's end, with the fragments it sent, nothing made up for the
    /// object the model never closed, and one `Malformed` event that carries what was held, a
    /// key, a value or a separator cut short, or no bytes ([`end_cut_call`]; the engine names the
    /// reason for a region a marker closed). A name that never got its `(` comes back as
    /// `Malformed`, and the wrapping before any call is dropped. The marker that closed the
    /// region is not passed: it belongs to the region, not to the last call, and the engine
    /// drops it.
    pub fn close(mut self, out: &mut Events) {
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Prose | Stage::Opening { .. } | Stage::Holding(_) => self.release_content(out),
            Stage::After => self.release_after(out),
            Stage::Between => self.drop_carried(out),
            Stage::Name { .. } => self.report(MalformedReason::NameWithoutACall, out),
            Stage::Arguments | Stage::Key { .. } | Stage::Value { .. } => {
                end_cut_call(self.index, std::mem::take(&mut self.carried), "", out);
            }
        }
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client. A call that started still ends, with no bytes of its own, as every assembler ends
    /// one and as [`Self::close`] ends a call the marker cut, after the one `Malformed {
    /// UnterminatedRegion }` event that says its arguments are no whole value
    /// ([`end_cut_call`]): it carries what was held, a key, a value or a separator cut short, or
    /// no bytes when the cut came right after the `(`, so that a call the end cut before its `)`
    /// is never handed over as finished without a word. Outside a call, what was held, a name
    /// that never got its `(` or the list's wrapping, comes back as `Malformed {
    /// UnterminatedRegion }` too. Under the bare list, content, a `[` no call followed yet and a
    /// list held after content come back as the content they are, and the text after the list as
    /// what it was.
    pub fn finish(mut self, out: &mut Events) {
        match self.stage {
            Stage::Prose | Stage::Opening { .. } | Stage::Holding(_) => {
                self.release_content(out);
                return;
            }
            Stage::After => {
                self.release_after(out);
                return;
            }
            _ => {}
        }
        let held = std::mem::take(&mut self.carried);
        if matches!(
            self.stage,
            Stage::Arguments | Stage::Key { .. } | Stage::Value { .. }
        ) {
            end_cut_call(self.index, held, "", out);
        } else if !held.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(held),
                why: MalformedReason::UnterminatedRegion,
            });
        }
    }

    fn take(&mut self, c: char, declared: &Declared, out: &mut Events) {
        match &mut self.stage {
            Stage::Prose => {
                if c == '[' {
                    if self.begun {
                        self.release_content(out);
                    }
                    self.carried.push('[');
                    self.stage = Stage::Opening {
                        named: false,
                        spaced: false,
                    };
                } else {
                    self.begun |= !c.is_whitespace();
                    self.carried.push(c);
                }
            }
            Stage::Opening { named, spaced } => {
                let (named, spaced) = (*named, *spaced);
                let opens = match c {
                    c if c.is_whitespace() => {
                        self.stage = Stage::Opening {
                            named,
                            spaced: named,
                        };
                        self.carried.push(c);
                        return;
                    }
                    '(' => named,
                    c if !named && starts_a_name(c) => {
                        self.stage = Stage::Opening {
                            named: true,
                            spaced: false,
                        };
                        self.carried.push(c);
                        return;
                    }
                    c if named && !spaced && continues_a_name(c) => {
                        self.carried.push(c);
                        return;
                    }
                    _ => false,
                };
                if opens && self.begun {
                    // A list after content: held until it is whole, then read if well formed. A
                    // first name no tool declares can only end as content (`[f(x)g'(x) + ...]`,
                    // `[Authorize(`), so it is content from its `(`, and the stream goes on.
                    let name = self.carried[1..].trim();
                    self.stage = if declared.declares(name) {
                        Stage::Holding(ListExtent::opened())
                    } else {
                        Stage::Prose
                    };
                } else if opens {
                    // The turn's list: the bytes held read as the list's, wrapping and name, and
                    // the `(` starts the call.
                    self.stage = Stage::Between;
                    let held = std::mem::take(&mut self.carried);
                    for held in held.chars() {
                        self.take(held, declared, out);
                    }
                } else {
                    // Not a list of calls: the `[` and what followed it are content, the turn
                    // has begun, and the character is read where content is.
                    self.begun = true;
                    self.stage = Stage::Prose;
                }
                self.take(c, declared, out);
            }
            Stage::Holding(extent) => {
                self.carried.push(c);
                match extent.read(c) {
                    // The backstop for a declared name followed by a stray quote: a hold that
                    // outgrows the cap (far above any real call's arguments) is content, and the
                    // stream goes on; an undeclared first name ended at its `(` already.
                    ListEnd::Open if self.carried.len() > HOLD_CAP_BYTES => {
                        self.stage = Stage::Prose;
                    }
                    ListEnd::Open => {}
                    ListEnd::Closed => self.read_held_list(declared, out),
                    ListEnd::Mismatched => self.stage = Stage::Prose,
                }
            }
            Stage::Between => match c {
                ']' if self.bare => {
                    self.carried.push(c);
                    self.drop_carried(out);
                    self.stage = Stage::After;
                }
                c if c.is_whitespace() || c == '[' || c == ']' || c == ',' => self.carried.push(c),
                c if starts_a_name(c) => {
                    self.drop_carried(out);
                    self.carried.push(c);
                    self.stage = Stage::Name { spaced: false };
                }
                c => {
                    self.drop_carried(out);
                    out.push(Event::Malformed {
                        text: Text::uncounted(c.to_string()),
                        why: MalformedReason::TextBetweenCalls,
                    });
                }
            },
            Stage::Name { spaced } => match c {
                c if !*spaced && continues_a_name(c) => {
                    self.carried.push(c);
                }
                // Python allows whitespace between the name and its `(`.
                c if c.is_whitespace() => {
                    *spaced = true;
                    self.carried.push(c);
                }
                '(' => {
                    let source = std::mem::take(&mut self.carried);
                    let name = source.trim_end().to_string();
                    self.index = self.next_index;
                    self.next_index += 1;
                    self.written = 0;
                    self.name.clone_from(&name);
                    out.push(Event::ToolCallStart {
                        index: self.index,
                        id: self.ids.id(self.index),
                        name,
                        source: Text::uncounted(format!("{source}(")),
                    });
                    self.stage = Stage::Arguments;
                }
                c => {
                    // Not a call after all: the name's bytes are reported, and the character is
                    // read where a call could begin.
                    self.report(MalformedReason::NameWithoutACall, out);
                    self.stage = Stage::Between;
                    self.take(c, declared, out);
                }
            },
            Stage::Arguments => match c {
                c if c.is_whitespace() => self.carried.push(c),
                ')' => {
                    let source = Text::uncounted(std::mem::take(&mut self.carried));
                    self.end_call(source, out);
                    self.stage = Stage::Between;
                }
                c if starts_a_name(c) => {
                    let start = self.carried.len();
                    self.carried.push(c);
                    self.stage = Stage::Key {
                        start,
                        spaced: false,
                    };
                }
                // A positional argument, a spread, or anything else where a key should begin.
                c => {
                    self.drop_carried(out);
                    out.push(Event::Malformed {
                        text: Text::uncounted(c.to_string()),
                        why: MalformedReason::TextWhereAnArgumentShouldBegin,
                    });
                }
            },
            Stage::Key { start, spaced } => match c {
                c if !*spaced && starts_a_name(c) => self.carried.push(c),
                // Python allows whitespace around the `=`; the value's extent trims its own.
                c if c.is_whitespace() => {
                    *spaced = true;
                    self.carried.push(c);
                }
                '=' => {
                    let key = self.carried[*start..].trim_end().to_string();
                    self.carried.push('=');
                    let start = self.carried.len();
                    self.stage = Stage::Value {
                        key,
                        start,
                        scan: Extent::default(),
                    };
                }
                c => {
                    self.report(MalformedReason::KeyWithoutAValue, out);
                    self.stage = Stage::Arguments;
                    self.take(c, declared, out);
                }
            },
            Stage::Value { key, start, scan } => {
                if scan.ends_at(c) {
                    let (key, start) = (key.clone(), *start);
                    self.write_value(&key, start, declared, out);
                    if c == ')' {
                        self.end_call(Text::uncounted(String::new()), out);
                        self.stage = Stage::Between;
                    } else {
                        self.carried.push(',');
                        self.stage = Stage::Arguments;
                    }
                } else {
                    self.carried.push(c);
                }
            }
            Stage::After => {
                let space = c.is_whitespace();
                let run_is_space = self.carried.chars().next().is_some_and(char::is_whitespace);
                if !self.carried.is_empty() && run_is_space != space {
                    self.release_after(out);
                }
                self.carried.push(c);
            }
        }
    }

    /// The list held after content closed: read whole by an assembler of its own, it is the calls
    /// it holds when every byte of it is a call's or the list's wrapping and every call names a
    /// tool the request declares, and content otherwise.
    fn read_held_list(&mut self, declared: &Declared, out: &mut Events) {
        let held = std::mem::take(&mut self.carried);
        let mut list = Assembler::new(
            self.next_index,
            self.ids,
            Dialect {
                framing: Framing::List,
                values: self.values,
            },
        );
        let mut events = Events::new();
        list.feed(&held, declared, &mut events);
        let next_index = list.next_index;
        list.close(&mut events);
        let events = events.drain();
        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let well_formed = !names.is_empty()
            && names.iter().all(|name| declared.declares(name))
            && !events
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. }));
        if well_formed {
            for event in events {
                out.push(event);
            }
            self.next_index = next_index;
            self.stage = Stage::After;
        } else {
            out.push(Event::Content(Text::uncounted(held)));
            self.stage = Stage::Prose;
        }
    }

    /// The value's extent is complete: `"key": <json>` with the key, the `=`, the value and the
    /// whitespace before them as its source. An extent with no value in it (`key=` right before
    /// the comma or the close) is a member Python refuses and the templates never write: its
    /// bytes are reported whole under `KeyWithoutAValue` and nothing is written for it, so the
    /// object closes whole without it and no empty string is made up for a value the model did
    /// not write.
    fn write_value(&mut self, key: &str, start: usize, declared: &Declared, out: &mut Events) {
        let text = self.carried[start..].trim().to_string();
        if text.is_empty() {
            return self.skip_member_without_a_value(out);
        }
        let value = match self.values {
            Values::Quoted => quoted_value(&text, declared.kind(&self.name, key)),
            Values::Python | Values::Json => python_literal(&text)
                .filter(|json| reads_back(json))
                .unwrap_or_else(|| json(&text, None)),
        };
        let mut fragment = String::with_capacity(key.len() + value.len() + 8);
        fragment.push_str(if self.written == 0 { "{" } else { ", " });
        fragment.push_str(&serde_json::Value::String(key.to_string()).to_string());
        fragment.push_str(": ");
        fragment.push_str(&value);
        self.written += 1;
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: fragment,
            source: Text::uncounted(std::mem::take(&mut self.carried)),
        });
    }

    /// The call's own `)`: `}` (or `{}`) with `source`, then `ToolCallEnd` with the parenthesis.
    /// It is the one end that closes a call's object; a region's marker or the stream's end
    /// never does.
    fn end_call<'d>(&mut self, source: Text<'d>, out: &mut Events<'d>) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: if self.written == 0 { "{}" } else { "}" }.to_string(),
            source,
        });
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(")"),
        });
    }

    /// `key=` with nothing before the comma or the close: the member's bytes come back as
    /// `Malformed { KeyWithoutAValue }` and nothing is written for it. Out of line: the templates
    /// never write the shape, and the value path stays as it is for the values they do write.
    #[cold]
    #[inline(never)]
    fn skip_member_without_a_value(&mut self, out: &mut Events) {
        self.report(MalformedReason::KeyWithoutAValue, out);
    }

    fn report(&mut self, why: MalformedReason, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Malformed {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why,
        });
    }

    fn drop_carried(&mut self, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Dropped {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: DropReason::Wrapper,
        });
    }

    /// The content held under the bare list, given back.
    fn release_content(&mut self, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Content(Text::uncounted(std::mem::take(
            &mut self.carried,
        ))));
    }

    /// The run held after the bare list, given back: whitespace as the template's wrapping,
    /// anything else as text after the list's end.
    fn release_after(&mut self, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        let text = Text::uncounted(std::mem::take(&mut self.carried));
        out.push(if text.text.chars().all(char::is_whitespace) {
            Event::Dropped {
                text,
                why: DropReason::Wrapper,
            }
        } else {
            Event::Malformed {
                text,
                why: MalformedReason::TextAfterTheList,
            }
        });
    }
}

/// Whether `json` is one JSON value a client can take as an argument.
fn reads_back(json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json).is_ok()
}

/// A value under [`Values::Quoted`]: between double quotes, the text inside them as it stands,
/// since the template writes the string's characters as they are and decodes no escape (a
/// backslash and an `r` in a value are those two characters, not a carriage return), typed by
/// `kind`, the one the request's tool declares for the parameter, or by its text where none is
/// declared; a value the model wrote bare, a single-quoted Python string included, reads as a
/// Python literal, and any other text as the tagged tables read theirs.
fn quoted_value(text: &str, kind: Option<Kind>) -> String {
    if let Some(inner) = between_double_quotes(text) {
        return json(inner, kind);
    }
    python_literal(text)
        .filter(|json| reads_back(json))
        .unwrap_or_else(|| json(text, kind))
}

/// The text between the double quotes of `text`, when it begins and ends with one.
fn between_double_quotes(text: &str) -> Option<&str> {
    text.strip_prefix('"')?.strip_suffix('"')
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize {
        self.feed(text, declared, out);
        text.len()
    }

    /// The region's marker belongs to the region, not to its last call.
    fn close(self: Box<Self>, _terminal: &str, out: &mut Events) -> bool {
        (*self).close(out);
        false
    }

    fn finish(self: Box<Self>, out: &mut Events) {
        (*self).finish(out);
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::testing::bytes;

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }

    fn run(pieces: &[&str]) -> Vec<Event<'static>> {
        run_with(Dialect::OLMO3, &Declared::default(), pieces)
    }

    fn run_with(dialect: Dialect, declared: &Declared, pieces: &[&str]) -> Vec<Event<'static>> {
        let mut assembler = Assembler::new(0, ids(), dialect);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, declared, &mut out);
        }
        assembler.close(&mut out);
        out.drain()
    }

    /// How a region ends: its closing marker (`close`) or the stream's end (`finish`).
    #[derive(Clone, Copy, Debug)]
    enum Ending {
        Close,
        Stream,
    }

    impl Ending {
        /// The Olmo 3 dialect's region after `pieces`, nothing declared, ended this way.
        fn run(self, pieces: &[&str]) -> Vec<Event<'static>> {
            let mut assembler = Assembler::new(0, ids(), Dialect::OLMO3);
            let mut out = Events::new();
            for piece in pieces {
                assembler.feed(piece, &Declared::default(), &mut out);
            }
            match self {
                Ending::Close => assembler.close(&mut out),
                Ending::Stream => assembler.finish(&mut out),
            }
            out.drain()
        }
    }

    fn dropped<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// Content with a `[name(` of its own: a name no tool declares is given back as content at its
    /// `(`, before the region ends, and the turn streams on.
    #[test]
    fn a_hold_after_content_whose_name_no_tool_declares_is_content_at_once() {
        let text = "The product rule gives [f(x)g'(x) + ";
        let mut assembler = Assembler::bare(0, ids(), Dialect::LLAMA4);
        let mut out = Events::new();
        assembler.feed(text, &tools(), &mut out);
        let while_open = out.drain();
        assert_eq!(
            content(&while_open),
            text,
            "given back before the region ends"
        );
        assembler.finish(&mut out);
        let mut events = while_open;
        events.extend(out.drain());
        assert_eq!(bytes(&events), text);
        assert!(calls(&events).is_empty());
        assert_eq!(malformed(&events), "");
    }

    /// A declared name after content with a lone quote after it: the hold would never close, so
    /// it is given back as content once it passes the cap, before the region ends, and the turn
    /// streams on.
    #[test]
    fn a_hold_after_content_is_given_back_once_it_passes_the_cap() {
        let text = format!(
            "Checking.[get_weather(city='Paris{}",
            "z".repeat(HOLD_CAP_BYTES)
        );
        let mut assembler = Assembler::bare(0, ids(), Dialect::LLAMA4);
        let mut out = Events::new();
        assembler.feed(&text, &tools(), &mut out);
        let while_open = out.drain();
        assert!(
            content(&while_open).len() > HOLD_CAP_BYTES,
            "given back before the region ends: {} bytes of content",
            content(&while_open).len()
        );
        assembler.finish(&mut out);
        let mut events = while_open;
        events.extend(out.drain());
        assert_eq!(bytes(&events), text);
        assert_eq!(content(&events), text);
        assert!(calls(&events).is_empty());
        assert_eq!(malformed(&events), "");
    }

    fn run_bare(declared: &Declared, pieces: &[&str]) -> Vec<Event<'static>> {
        let mut assembler = Assembler::bare(0, ids(), Dialect::LLAMA4);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, declared, &mut out);
        }
        assembler.finish(&mut out);
        out.drain()
    }

    fn calls(events: &[Event]) -> Vec<(u32, String, String)> {
        let mut calls: Vec<(u32, String, String)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { index, name, .. } => {
                    calls.push((*index, name.clone(), String::new()));
                }
                Event::ToolCallArguments { json, .. } => {
                    if let Some(last) = calls.last_mut() {
                        last.2.push_str(json);
                    }
                }
                _ => {}
            }
        }
        calls
    }

    fn content(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The text of every `Malformed` event, one entry per event, whatever its reason: an event of
    /// no bytes counts here, where the joined text of [`malformed`] does not show it.
    fn words<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The malformed text, whole: the events that carry it depend on the cuts, the text does not.
    fn malformed(events: &[Event]) -> String {
        words(events).concat()
    }

    fn tool(name: &str, parameters: serde_json::Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        }
    }

    fn tools() -> Declared {
        Declared::of(&[
            tool(
                "get_weather",
                value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                    "days": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "detailed": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "prefs": {"type": "object"},
                    "note": {"type": ["string", "null"]},
                }}),
            ),
            tool(
                "get_stock_price",
                value!({"type": "object", "properties": {}}),
            ),
            tool(
                "save_note",
                value!({"type": "object", "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                }}),
            ),
        ])
    }

    const OLMO: &str = concat!(
        "get_weather(city=\"Paris\", days=3, ",
        "prefs={\"size\": \"large\", \"deep\": [1, 2.5, true, null]})\n",
        "get_stock_price()"
    );
    const OLMO_ARGUMENTS: &str = concat!(
        r#"{"city": "Paris", "days": 3, "#,
        r#""prefs": {"size": "large", "deep": [1, 2.5, true, null]}}"#
    );
    const LFM: &str = "[get_weather(city='Paris', days=3, note='it\\'s fine'), get_stock_price()]";

    #[test]
    fn olmos_bare_calls_give_two_calls_with_their_values_and_every_byte() {
        let events = run(&[OLMO]);
        assert_eq!(bytes(&events), OLMO);
        assert_eq!(
            calls(&events),
            [
                (0, "get_weather".to_string(), OLMO_ARGUMENTS.to_string()),
                (1, "get_stock_price".to_string(), "{}".to_string()),
            ]
        );
        assert_eq!(
            events[0],
            Event::ToolCallStart {
                index: 0,
                id: ids().id(0),
                name: "get_weather".into(),
                source: Text::uncounted("get_weather("),
            }
        );
    }

    #[test]
    fn lfms_list_gives_the_same_calls_with_the_brackets_and_commas_dropped() {
        let events = run_with(Dialect::LFM2_5, &Declared::default(), &[LFM]);
        assert_eq!(bytes(&events), LFM);
        assert_eq!(
            calls(&events),
            [
                (
                    0,
                    "get_weather".to_string(),
                    r#"{"city": "Paris", "days": 3, "note": "it's fine"}"#.to_string()
                ),
                (1, "get_stock_price".to_string(), "{}".to_string()),
            ]
        );
        assert_eq!(dropped(&events), ["[", ", ", "]"]);
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        for text in [OLMO, LFM] {
            let whole = calls(&run(&[text]));
            for cut in 1..text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let events = run(&[&text[..cut], &text[cut..]]);
                assert_eq!(bytes(&events), text, "cut at {cut}");
                assert_eq!(calls(&events), whole, "cut at {cut}");
            }
        }
    }

    #[test]
    fn a_call_either_ending_cuts_inside_a_value_closes_nothing_and_says_so() {
        for ending in [Ending::Close, Ending::Stream] {
            let events = ending.run(&["get_weather(city=\"Par"]);
            assert_eq!(bytes(&events), "get_weather(city=\"Par", "{ending:?}");
            // The call ends with no bytes of its own and no fragment: its one word carries the
            // value cut short.
            assert!(
                matches!(
                    &events[..],
                    [
                        Event::ToolCallStart { index: 0, .. },
                        Event::Malformed { why: MalformedReason::UnterminatedRegion, .. },
                        Event::ToolCallEnd { index: 0, source },
                    ] if source.text.is_empty()
                ),
                "{ending:?}: {events:?}"
            );
            assert_eq!(said(&events), ["city=\"Par"], "{ending:?}");
        }
    }

    #[test]
    fn prose_where_a_call_should_begin_is_malformed_and_the_calls_around_it_still_come() {
        let text = "I will call:\nget_weather(city=\"Paris\")";
        let events = run(&[text]);
        assert_eq!(bytes(&events), text);
        assert_eq!(calls(&events).len(), 1);
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::Malformed { .. })));
    }

    #[test]
    fn pythons_whitespace_around_the_parenthesis_and_the_equals_sign_is_allowed() {
        let output = "get_weather (city = \"Paris\", days= 3 )";
        let events = run(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [(
                0,
                "get_weather".to_string(),
                r#"{"city": "Paris", "days": 3}"#.to_string()
            )]
        );
    }

    /// A parameter whose value is missing before the comma or the close is a member Python
    /// refuses and the templates never write: its bytes, the separator before it, the key and
    /// the `=` with the whitespace around them, come back under the one reason, nothing is
    /// written for it, and the object
    /// closes whole without it, whole and by the character alike; no empty string stands for the
    /// value the model did not write.
    #[test]
    fn a_parameter_without_a_value_is_reported_and_skipped_and_the_object_closes_whole() {
        let cases: &[(&str, &str, &[&str])] = &[
            ("[get_weather(city=, days=3)]", r#"{"days": 3}"#, &["city="]),
            ("[get_weather(city=)]", "{}", &["city="]),
            (
                "[get_weather(city= , days=3)]",
                r#"{"days": 3}"#,
                &["city= "],
            ),
            (
                "[get_weather(city='Paris', days=)]",
                r#"{"city": "Paris"}"#,
                &[", days="],
            ),
            (
                "[get_weather(city='Paris', days= )]",
                r#"{"city": "Paris"}"#,
                &[", days= "],
            ),
            ("[get_weather(a=, b=)]", "{}", &["a=", ", b="]),
            ("get_weather(city=, days=3)\nget_stock_price(q=)", "", &[]),
        ];
        for (text, arguments, reported) in cases {
            let pieces: Vec<String> = text.chars().map(String::from).collect();
            let whole = run(&[text]);
            let by_char = run(&pieces.iter().map(String::as_str).collect::<Vec<_>>());
            for events in [whole, by_char] {
                assert_eq!(bytes(&events), *text);
                let words: Vec<&str> = events
                    .iter()
                    .filter_map(|event| match event {
                        Event::Malformed {
                            text,
                            why: MalformedReason::KeyWithoutAValue,
                        } => Some(&*text.text),
                        _ => None,
                    })
                    .collect();
                if arguments.is_empty() {
                    // Olmo's bare calls, one per line: each call keeps its own members.
                    assert_eq!(
                        calls(&events),
                        [
                            (0, "get_weather".to_string(), r#"{"days": 3}"#.to_string()),
                            (1, "get_stock_price".to_string(), "{}".to_string()),
                        ],
                        "{text}"
                    );
                    assert_eq!(words, ["city=", "q="], "{text}");
                } else {
                    assert_eq!(
                        calls(&events),
                        [(0, "get_weather".to_string(), arguments.to_string())],
                        "{text}"
                    );
                    assert_eq!(words, *reported, "{text}");
                }
                for (_, _, arguments) in calls(&events) {
                    assert!(
                        serde_json::from_str::<serde_json::Value>(&arguments).is_ok(),
                        "{text}: {arguments} is whole"
                    );
                }
                assert!(
                    !events.iter().any(|event| matches!(
                        event,
                        Event::ToolCallArguments { json, .. } if json.contains("\"\"")
                    )),
                    "{text}: no empty string is written"
                );
            }
        }
    }

    #[test]
    fn a_positional_argument_is_reported_as_text_where_an_argument_should_begin() {
        let events = run(&["get_weather(\"Paris\", days=3)"]);
        assert_eq!(
            calls(&events),
            [(0, "get_weather".to_string(), r#"{"days": 3}"#.to_string())]
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::TextWhereAnArgumentShouldBegin,
                ..
            }
        )));
    }

    // The quoted dialect: every value between double quotes, typed by the request's tools.

    const QUOTED: &str = concat!(
        "[get_weather(city=\"Paris\", zip=\"75001\", days=\"3\", ratio=\"2.5\", ",
        "detailed=\"True\", tags=\"['a', 'b']\", ",
        "prefs=\"{'size': 'large', 'deep': [1, 2.5, True, None]}\", note=\"None\")]"
    );
    const QUOTED_ARGUMENTS: &str = concat!(
        r#"{"city": "Paris", "zip": "75001", "days": 3, "ratio": 2.5, "detailed": true, "#,
        r#""tags": ["a", "b"], "prefs": {"size": "large", "deep": [1, 2.5, true, null]}, "#,
        r#""note": null}"#
    );

    #[test]
    fn a_quoted_value_reads_by_the_kind_its_tool_declares() {
        let events = run_bare(&tools(), &[QUOTED]);
        assert_eq!(bytes(&events), QUOTED);
        assert_eq!(
            calls(&events),
            [(0, "get_weather".to_string(), QUOTED_ARGUMENTS.to_string())]
        );
        assert_eq!(malformed(&events), "");
        // Without a declaration the text decides: a number, a word and a literal read as such,
        // and a word that is none of them is the string.
        let events = run_bare(
            &Declared::default(),
            &["[f(zip=\"75001\", city=\"Paris\", on=\"False\", tags=\"[1, 2]\")]"],
        );
        assert_eq!(
            calls(&events),
            [(
                0,
                "f".to_string(),
                r#"{"zip": 75001, "city": "Paris", "on": false, "tags": [1, 2]}"#.to_string()
            )]
        );
        // A value the model writes bare reads as Python's, whatever the tool declares, a
        // single-quoted string with its escape; a double-quoted one is the template's, read as
        // written, backslashes and all.
        let events = run_bare(
            &tools(),
            &["[get_weather(days=3, detailed=True, city='it\\'s', zip=\"a \\\"q\\\"\")]"],
        );
        assert_eq!(
            calls(&events),
            [(
                0,
                "get_weather".to_string(),
                r#"{"days": 3, "detailed": true, "city": "it's", "zip": "a \\\"q\\\""}"#
                    .to_string()
            )]
        );
        // A string the template wrote as it stands, with a newline, a backslash and the two
        // characters `\r` in it, is the string, characters and all.
        let events = run_bare(
            &tools(),
            &["[get_weather(city=\"Line one\nLine two\", zip=\"C:\\Users\\x\\r\")]"],
        );
        assert_eq!(
            calls(&events),
            [(
                0,
                "get_weather".to_string(),
                r#"{"city": "Line one\nLine two", "zip": "C:\\Users\\x\\r"}"#.to_string()
            )]
        );
    }

    #[test]
    fn the_bare_list_opens_at_the_turns_start_alone_and_anything_else_is_content() {
        let text = "\n [get_weather(city=\"Paris\"), get_stock_price()]";
        let events = run_bare(&tools(), &[text]);
        assert_eq!(bytes(&events), text);
        assert_eq!(content(&events), "");
        assert_eq!(
            calls(&events),
            [
                (
                    0,
                    "get_weather".to_string(),
                    r#"{"city": "Paris"}"#.to_string()
                ),
                (1, "get_stock_price".to_string(), "{}".to_string()),
            ]
        );
        assert_eq!(malformed(&events), "");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "\n ["
        )));
        // After content, a list reads once it is whole and well formed: the shape the template
        // renders for a message with both.
        let text = "Let me check.[get_weather(city=\"Paris\"), get_stock_price()]";
        let events = run_bare(&tools(), &[text]);
        assert_eq!(bytes(&events), text);
        assert_eq!(content(&events), "Let me check.");
        assert_eq!(calls(&events).len(), 2);
        assert_eq!(malformed(&events), "");
        // Code holds `[name(` of its own: a positional argument, an operator, a nested call, a
        // name no tool declares make the held list content; so does a list the stream's end cut.
        for prose in [
            "f'(a) = lim(h→0)[f(a+h) - f(a)]/h",
            "const m = movies[Math.floor(Math.random() * movies.length)];",
            "model = Sequential([\n tf.keras.layers.Dense(64, activation='relu'),\n])",
            "json.dumps([float(num) for num in string_list])",
            "ingredients[exotic_ingredient.lower()] = {'amount': '8 ounces'}",
            "// managers only\n[Authorize(Roles = \"Manager\")]\npublic ActionResult Feedback()",
            "Let me check.[get_weather(city=\"Par",
            "Let me check.[get_weather(city=\"Paris\")",
            "a[get_weather(city=\"Paris\")} b",
            "[Note] see [1, 2] and [a link](https://example.com) or [x] [y z] [].",
            "The list [get_weather, get_stock_price] names two tools.",
            "[",
            "[get_weather",
            "  ",
            "",
        ] {
            let events = run_bare(&tools(), &[prose]);
            assert_eq!(bytes(&events), prose, "{prose:?}");
            assert_eq!(content(&events), prose, "{prose:?}");
            assert!(calls(&events).is_empty(), "{prose:?}");
            assert_eq!(malformed(&events), "", "{prose:?}");
        }
        // A member without a value, at both positions: at the turn's start the list streams as it
        // arrives, so the call comes with the member reported and left out; after content a held
        // list is read as calls only when it is well-formed Python, and `city=` is not, so the
        // whole is content, as any held list with a malformed byte is.
        let text = "[get_weather(city=, days=3)]";
        let events = run_bare(&tools(), &[text]);
        assert_eq!(bytes(&events), text);
        assert_eq!(content(&events), "");
        assert_eq!(
            calls(&events),
            [(0, "get_weather".to_string(), r#"{"days": 3}"#.to_string())]
        );
        assert_eq!(malformed(&events), "city=");
        let text = "Let me check.[get_weather(city=, days=3)]";
        let events = run_bare(&tools(), &[text]);
        assert_eq!(bytes(&events), text);
        assert_eq!(content(&events), text);
        assert!(calls(&events).is_empty());
        assert_eq!(malformed(&events), "");
        // Whitespace between the bracket and the name, and after the name, is Python's.
        let events = run_bare(&tools(), &["[ get_weather (city=\"Paris\") ]"]);
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(content(&events), "");
    }

    #[test]
    fn text_after_the_bare_list_is_wrapping_or_malformed_and_a_cut_list_stays_cut() {
        let text = "[get_weather(city=\"Paris\")]\n\nDone.\n";
        let events = run_bare(&tools(), &[text]);
        assert_eq!(bytes(&events), text);
        assert_eq!(content(&events), "");
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(malformed(&events), "Done.");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::TextAfterTheList,
                ..
            }
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "\n\n"
        )));
        // Cut inside a value: the call ends with nothing of its own, and the value, held until
        // its extent is known, comes back as the unterminated region's text.
        let events = run_bare(&tools(), &["[get_weather(city=\"Par"]);
        assert_eq!(content(&events), "");
        assert_eq!(
            calls(&events),
            [(0, "get_weather".to_string(), String::new())]
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::UnterminatedRegion }
                if text.text == "city=\"Par"
        )));
    }

    #[test]
    fn the_bare_list_says_the_same_at_every_cut() {
        for text in [
            QUOTED,
            "Let me check.[get_weather(city=\"Paris\"), get_stock_price()]",
            "Sure.\n\n[save_note(title=\"a]b\", body=\"(x)\")]\n",
            "\n [get_weather(city=\"Paris\"), get_stock_price()]",
            "f'(a) = lim(h→0)[f(a+h) - f(a)]/h",
            "[Note] see [1, 2] and [a link](https://example.com).",
            "[get_weather(city=\"Paris\")]\n\nDone.",
            "[get_weather(city=, days=3)]",
            "Let me check.[get_weather(city=, days=3)]",
        ] {
            let whole = run_bare(&tools(), &[text]);
            for cut in 1..text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let events = run_bare(&tools(), &[&text[..cut], &text[cut..]]);
                assert_eq!(bytes(&events), text, "cut at {cut}");
                assert_eq!(calls(&events), calls(&whole), "cut at {cut}");
                assert_eq!(content(&events), content(&whole), "cut at {cut}");
                assert_eq!(malformed(&events), malformed(&whole), "cut at {cut}");
            }
            let pieces: Vec<&str> = text
                .char_indices()
                .map(|(at, c)| &text[at..at + c.len_utf8()])
                .collect();
            let events = run_bare(&tools(), &pieces);
            assert_eq!(bytes(&events), text, "character by character");
            assert_eq!(calls(&events), calls(&whole), "character by character");
            assert_eq!(content(&events), content(&whole), "character by character");
        }
    }

    fn by_the_character(text: &str) -> Vec<String> {
        text.chars().map(String::from).collect()
    }

    /// The words the end said: the text of every `Malformed { UnterminatedRegion }` event.
    fn said<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed {
                    text,
                    why: MalformedReason::UnterminatedRegion,
                } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_call_either_ending_cuts_right_after_its_parenthesis_says_so_before_its_end() {
        let start = |index: u32, name: &str| Event::ToolCallStart {
            index,
            id: ids().id(index),
            name: name.to_string(),
            source: Text::uncounted(format!("{name}(")),
        };
        let word = Event::Malformed {
            text: Text::uncounted(""),
            why: MalformedReason::UnterminatedRegion,
        };
        let end = |index: u32| Event::ToolCallEnd {
            index,
            source: Text::uncounted(""),
        };
        for ending in [Ending::Close, Ending::Stream] {
            assert_eq!(
                ending.run(&["[get_weather("]),
                [
                    Event::Dropped {
                        text: Text::uncounted("["),
                        why: DropReason::Wrapper,
                    },
                    start(0, "get_weather"),
                    word.clone(),
                    end(0),
                ],
                "{ending:?}"
            );
            // The same by the character, for a second call cut there: the first stands whole.
            let text = "[get_weather(city='Paris'), get_stock_price(";
            let pieces = by_the_character(text);
            let events = ending.run(&pieces.iter().map(String::as_str).collect::<Vec<_>>());
            assert_eq!(bytes(&events), text, "{ending:?}");
            assert_eq!(
                calls(&events),
                [
                    (
                        0,
                        "get_weather".to_string(),
                        r#"{"city": "Paris"}"#.to_string()
                    ),
                    (1, "get_stock_price".to_string(), String::new()),
                ],
                "{ending:?}"
            );
            assert_eq!(
                events[events.len() - 3..],
                [start(1, "get_stock_price"), word.clone(), end(1)],
                "{ending:?}"
            );
        }
    }

    /// What an ending finds at the cut: a call open (the one word carries what was held), whole
    /// calls alone, the list's wrapping after a whole call, or a name that never got its `(`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Found {
        Open,
        Whole,
        Wrapping,
        Name,
    }

    #[test]
    fn either_ending_says_so_wherever_it_cuts_a_call_and_carries_what_was_held() {
        // Where the ending cuts, what was held there, and what it finds.
        let cuts: &[(&str, &str, Found)] = &[
            ("[get_weather", "get_weather", Found::Name),
            ("[get_weather(", "", Found::Open),
            ("[get_weather( ", " ", Found::Open),
            ("[get_weather(ci", "ci", Found::Open),
            ("[get_weather(city ", "city ", Found::Open),
            ("[get_weather(city=", "city=", Found::Open),
            ("[get_weather(city='Par", "city='Par", Found::Open),
            ("[get_weather(city='Paris'", "city='Paris'", Found::Open),
            ("[get_weather(city='Paris',", ",", Found::Open),
            ("[get_weather(city='Paris', ", ", ", Found::Open),
            ("[get_weather(city='Paris', da", ", da", Found::Open),
            (
                "[get_weather(prefs={'a': [1, ",
                "prefs={'a': [1, ",
                Found::Open,
            ),
            ("[get_weather(city='Paris')", "", Found::Whole),
            ("[get_weather(city='Paris'), ", ", ", Found::Wrapping),
            (
                "[get_weather(city='Paris'), get_stock_price(",
                "",
                Found::Open,
            ),
        ];
        for ending in [Ending::Close, Ending::Stream] {
            for (text, held, found) in cuts {
                let pieces = by_the_character(text);
                let whole = ending.run(&[text]);
                let by_the_character =
                    ending.run(&pieces.iter().map(String::as_str).collect::<Vec<_>>());
                for events in [whole, by_the_character] {
                    let place = format!("{ending:?} after {text:?}");
                    assert_eq!(bytes(&events), *text, "{place}");
                    let starts = events
                        .iter()
                        .filter(|event| matches!(event, Event::ToolCallStart { .. }))
                        .count();
                    let ends = events
                        .iter()
                        .filter(|event| matches!(event, Event::ToolCallEnd { .. }))
                        .count();
                    assert_eq!(starts, ends, "{place}: every call that started ends");
                    match (found, ending) {
                        (Found::Open, _) => {
                            assert_eq!(said(&events), [*held], "{place}: one word, what was held");
                            assert!(
                                matches!(
                                    &events[events.len() - 2..],
                                    [Event::Malformed { .. }, Event::ToolCallEnd { source, .. }]
                                        if source.text.is_empty()
                                ),
                                "{place}: the word comes right before an end of no bytes"
                            );
                            // Nothing is made up: a cut call's fragments never close its object,
                            // so they are no whole JSON value (a member's own `}` may end them).
                            assert!(
                                calls(&events).last().is_some_and(|(_, _, arguments)| {
                                    serde_json::from_str::<serde_json::Value>(arguments).is_err()
                                }),
                                "{place}: the cut call's arguments stay open"
                            );
                        }
                        (Found::Whole, _) => {
                            assert!(words(&events).is_empty(), "{place}: no word");
                        }
                        // The wrapping after a whole call is the template's at the close and a
                        // cut region's bytes at the stream's end.
                        (Found::Wrapping, Ending::Close) => {
                            assert!(words(&events).is_empty(), "{place}: no word");
                            assert!(dropped(&events).contains(held), "{place}: wrapping dropped");
                        }
                        (Found::Wrapping, Ending::Stream) => {
                            assert_eq!(said(&events), [*held], "{place}: the held wrapping");
                        }
                        // A name that never got its `(` is no call either way, and the close
                        // names its reason.
                        (Found::Name, Ending::Close) => {
                            assert_eq!(words(&events), [*held], "{place}: the name");
                            assert!(said(&events).is_empty(), "{place}: the close's own reason");
                        }
                        (Found::Name, Ending::Stream) => {
                            assert_eq!(said(&events), [*held], "{place}: the name");
                        }
                    }
                    if *found != Found::Open {
                        assert!(
                            calls(&events)
                                .iter()
                                .all(|(_, _, arguments)| arguments.ends_with('}')),
                            "{place}: every call is whole"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_call_either_ending_cuts_anywhere_keeps_the_fragments_it_sent_and_nothing_more() {
        let whole = calls(&run(&[LFM]));
        for (ending, at) in [Ending::Close, Ending::Stream]
            .into_iter()
            .flat_map(|ending| (1..LFM.len()).map(move |at| (ending, at)))
        {
            if !LFM.is_char_boundary(at) {
                continue;
            }
            let text = &LFM[..at];
            let events = ending.run(&[text]);
            let place = format!("{ending:?} at {at}");
            assert_eq!(bytes(&events), text, "{place}");
            for (index, name, arguments) in calls(&events) {
                let (_, whole_name, whole_arguments) = &whole[index as usize];
                assert_eq!(&name, whole_name, "{place}");
                // A call's fragments are a prefix of what the whole call gives, and the one word
                // stands between a cut call's start and its end, never inside a whole one.
                assert!(
                    whole_arguments.starts_with(&arguments),
                    "{place}: {arguments:?} is no prefix of {whole_arguments:?}"
                );
                let start = events
                    .iter()
                    .position(|event| matches!(event, Event::ToolCallStart { index: i, .. } if *i == index))
                    .unwrap_or_else(|| panic!("{place}: call {index} has no start"));
                let end = events
                    .iter()
                    .position(
                        |event| matches!(event, Event::ToolCallEnd { index: i, .. } if *i == index),
                    )
                    .unwrap_or_else(|| panic!("{place}: call {index} has no end"));
                assert_eq!(
                    said(&events[start..end]).len(),
                    usize::from(&arguments != whole_arguments),
                    "{place}: a cut call says so once, a whole one not at all"
                );
            }
        }
    }
}
