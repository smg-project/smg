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
//! whole and well formed; after the list whitespace alone is the template's wrapping, and prose
//! is the model's, whitespace included.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    ids::CallIds,
    tagged::{
        assembler::push_escaped,
        end_cut_call,
        value::{cut as cut_json, json, python_literal, Kind},
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
    /// among them, so the next `"` ends the value whatever stands before it (a value that ends
    /// in a backslash closes at its quote), save that a `"` right after a backslash is the model's
    /// own escaped quote (`5\" screen`, which the family's own readers decode) unless the value's
    /// end follows it. A value between double quotes reads as written, by
    /// the kind the request's tool declares for its parameter, by its text where none is
    /// declared; a value the model writes bare, a single-quoted string included and the strings
    /// inside a bare list or object, reads as Python's, escapes and all.
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
    /// Under the bare list: whether text followed the list's `]`. From there on every byte,
    /// whitespace included, is the model's prose after its calls, which the template's reference
    /// keeps as written; before it, whitespace is held, the template's wrapping if nothing
    /// follows.
    prose_after: bool,
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
    /// Inside a value: its text is `carried[start..]`, scanned for its extent; `kind` is what the
    /// request's tool declares for the parameter, read where the dialect's quotes carry no type.
    Value {
        key: String,
        start: usize,
        scan: Extent,
        kind: Option<Kind>,
    },
    /// Under the bare list: after the list's `]`. `carried` holds whitespace nothing has followed
    /// yet, the template's wrapping if nothing does, or the prose's next word with the whitespace
    /// before it, which leaves when the word ends (`prose_after`).
    After,
}

/// Where a value ends: at a `,` or `)` outside every string and bracket.
#[derive(Clone, Copy, Debug)]
struct Extent {
    depth: u32,
    quote: Option<char>,
    escaped: bool,
    /// Under [`Values::Quoted`]: a double-quoted string at the value's top level is the
    /// template's text as it stands, so no backslash in it escapes the `"` that ends it; a string
    /// inside a list or an object the model writes bare is Python's, escapes and all.
    verbatim: bool,
    /// Whether a backslash escapes the next character in the open string: Python's rule, off in
    /// the template's own quotes, where `escaped` only remembers that a backslash came last.
    escapes: bool,
    /// In the template's own quotes: a `"` came right after a backslash, which is the value's end
    /// if `,` or `)` follows it, whitespace aside, and the model's own escaped quote otherwise,
    /// after which the string goes on.
    tentative: bool,
}

impl Extent {
    /// The extent of a value under `values`, before its first character.
    fn new(values: Values) -> Self {
        Self {
            depth: 0,
            quote: None,
            escaped: false,
            verbatim: values == Values::Quoted,
            escapes: true,
            tentative: false,
        }
    }

    /// Reads one character and says whether it ends the value (it is then not the value's).
    fn ends_at(&mut self, c: char) -> bool {
        if let Some(quote) = self.quote {
            if self.tentative {
                if c.is_whitespace() {
                    return false;
                }
                self.tentative = false;
                if c == ',' || c == ')' {
                    self.quote = None;
                    return true;
                }
                // The quote was the model's own escaped one: the string goes on, and `c` is its.
                self.escaped = c == '\\';
                if c == quote {
                    self.quote = None;
                }
            } else if self.escaped {
                self.escaped = false;
                if !self.escapes {
                    if c == quote {
                        self.tentative = true;
                    } else if c == '\\' {
                        self.escaped = true;
                    }
                }
            } else if c == '\\' {
                self.escaped = true;
            } else if c == quote {
                self.quote = None;
            }
            return false;
        }
        match c {
            '\'' | '"' => {
                self.quote = Some(c);
                self.escapes = !(self.verbatim && c == '"' && self.depth == 0);
            }
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
    /// As [`Extent::verbatim`]: a double-quoted string at a value's top level, inside the list's
    /// `[` and the call's `(` and nothing else, takes no escape under [`Values::Quoted`].
    verbatim: bool,
    /// As [`Extent::escapes`].
    escapes: bool,
    /// As [`Extent::tentative`].
    tentative: bool,
}

impl ListExtent {
    /// The extent after the list's `[`, under `values`.
    fn opened(values: Values) -> Self {
        Self {
            depth: 1,
            quote: None,
            escaped: false,
            verbatim: values == Values::Quoted,
            escapes: true,
            tentative: false,
        }
    }

    /// Reads one character and says what it leaves: the list still open, closed by its `]`, or
    /// closed by another bracket, which no list of calls writes.
    fn read(&mut self, c: char) -> ListEnd {
        if let Some(quote) = self.quote {
            if self.tentative {
                if c.is_whitespace() {
                    return ListEnd::Open;
                }
                self.tentative = false;
                if c == ',' || c == ')' {
                    // The value's end: the string is closed, and the character reads outside it.
                    self.quote = None;
                    return self.read(c);
                }
                self.escaped = c == '\\';
                if c == quote {
                    self.quote = None;
                }
            } else if self.escaped {
                self.escaped = false;
                if !self.escapes {
                    if c == quote {
                        self.tentative = true;
                    } else if c == '\\' {
                        self.escaped = true;
                    }
                }
            } else if c == '\\' {
                self.escaped = true;
            } else if c == quote {
                self.quote = None;
            }
            return ListEnd::Open;
        }
        match c {
            '\'' | '"' => {
                self.quote = Some(c);
                self.escapes = !(self.verbatim && c == '"' && self.depth == 2);
            }
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
            prose_after: false,
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
    /// end cut included, is content. After the list's `]` whitespace alone is the template's
    /// wrapping; once text follows, every byte after the `]` is the model's prose, whitespace
    /// included, reported malformed as it is after xLAM's bare list. A `[` that no call follows
    /// (`[Note]`, `[1, 2]`, a markdown link) is content too.
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
        // Content and the words after a bare list are given back as they arrive; a `[` under
        // consideration and a list held after content stay held, and so do the whitespace before
        // the turn's first byte, which a list would make its wrapping, and the whitespace after
        // the list, which the next word makes the prose's and the end the template's.
        match self.stage {
            Stage::Prose if self.begun => self.release_content(out),
            Stage::After if self.prose_after && !self.carried.ends_with(char::is_whitespace) => {
                self.release_after(out);
            }
            _ => {}
        }
    }

    /// The region's end: its closing marker, or the next opener. A call still open is one the
    /// model never finished, since every template writes the call's `)` before the marker: it
    /// ends with the fragments it sent, nothing made up for the object the model never closed,
    /// and one `Malformed` event that carries what was held, a key, a value or a separator cut
    /// short, or no bytes ([`end_cut_call`]; the engine names the reason for a region a marker
    /// closed); the stream's end (a cut by the token limit, an abort, or the model's own stop,
    /// which ends a pythonic call as a cut does, since its quotes and braces are the model's own)
    /// writes the held member as the call's own instead ([`Self::finish`]). A name that never got
    /// its `(` comes back as
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
    /// client, and the call's bytes stay the call's. A call that started still ends, with no
    /// bytes of its own, as every assembler ends one, after the one `Malformed {
    /// UnterminatedRegion }` event of no bytes that says its arguments are no whole value
    /// ([`end_cut_call`]), so that a call the end cut before its `)` is never handed over as
    /// finished without a word. Before it, the member the cut found inside its value leaves as
    /// the call's own fragment, its key and the JSON the assembler would have written for the
    /// value as far as that is known ([`Self::cut_value`]), since a client reads the prefix as
    /// written under `length`, never the call's bytes as the answer's prose; a key the cut found
    /// open, the separator and the whitespace after a member leave as the syntax's wrapping
    /// (`Dropped`), as a parameter tag the end cut short does under a tagged syntax, since no
    /// member can be written for them. Outside a call, what was held, a name that never got its
    /// `(` or the list's wrapping, comes back as `Malformed { UnterminatedRegion }`. Under the
    /// bare list, content, a `[` no call followed yet and a list held after content come back as
    /// the content they are, and the text after the list as what it was. The stream's end is a
    /// cut by the token limit, an abort, or the model's own stop, which ends a pythonic call as a
    /// cut does since its quotes and braces are the model's own (nothing closes for it either
    /// way); the region's marker inside a call ([`Self::close`]) still reports what was held.
    pub fn finish(mut self, out: &mut Events) {
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Prose | Stage::Opening { .. } | Stage::Holding(_) => self.release_content(out),
            Stage::After => self.release_after(out),
            Stage::Value {
                key,
                start,
                scan,
                kind,
            } => {
                self.cut_value(&key, start, scan, kind, out);
                end_cut_call(self.index, String::new(), "", out);
            }
            Stage::Arguments | Stage::Key { .. } => {
                self.drop_carried(out);
                end_cut_call(self.index, String::new(), "", out);
            }
            Stage::Between | Stage::Name { .. } => {
                if !self.carried.is_empty() {
                    out.push(Event::Malformed {
                        text: Text::uncounted(std::mem::take(&mut self.carried)),
                        why: MalformedReason::UnterminatedRegion,
                    });
                }
            }
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
                        Stage::Holding(ListExtent::opened(self.values))
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
                    // The quotes of [`Values::Quoted`] carry no type: the tool's declaration does.
                    let kind = (self.values == Values::Quoted)
                        .then(|| declared.kind(&self.name, &key))
                        .flatten();
                    self.carried.push('=');
                    let start = self.carried.len();
                    self.stage = Stage::Value {
                        key,
                        start,
                        scan: Extent::new(self.values),
                        kind,
                    };
                }
                c => {
                    self.report(MalformedReason::KeyWithoutAValue, out);
                    self.stage = Stage::Arguments;
                    self.take(c, declared, out);
                }
            },
            Stage::Value {
                key,
                start,
                scan,
                kind,
            } => {
                if scan.ends_at(c) {
                    let (key, start, kind) = (key.clone(), *start, *kind);
                    self.write_value(&key, start, kind, out);
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
                // Whitespace is held until what follows says whose it is; a word leaves where it
                // ends, with the whitespace before it, once text followed the list.
                if !c.is_whitespace() {
                    self.prose_after = true;
                } else if self.prose_after && !self.carried.ends_with(char::is_whitespace) {
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
    fn write_value(&mut self, key: &str, start: usize, kind: Option<Kind>, out: &mut Events) {
        let text = self.carried[start..].trim().to_string();
        if text.is_empty() {
            return self.skip_member_without_a_value(out);
        }
        let value = match self.values {
            Values::Quoted => quoted_value(&text, kind),
            Values::Python | Values::Json => python_literal(&text)
                .filter(|json| reads_back(json))
                .unwrap_or_else(|| json(&text, None)),
        };
        self.push_member(key, &value, out);
    }

    /// The member `"key": value` after `{` or `, `, counted as written, with every byte carried
    /// as its source.
    fn push_member(&mut self, key: &str, value: &str, out: &mut Events) {
        let mut fragment = String::with_capacity(key.len() + value.len() + 8);
        fragment.push_str(if self.written == 0 { "{" } else { ", " });
        fragment.push_str(&serde_json::Value::String(key.to_string()).to_string());
        fragment.push_str(": ");
        fragment.push_str(value);
        self.written += 1;
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: fragment,
            source: Text::uncounted(std::mem::take(&mut self.carried)),
        });
    }

    /// The member the end of the stream cut inside its value ([`Self::finish`]), written as far
    /// as the JSON the assembler writes for it can be known from the bytes so far, with the key,
    /// the `=`, the value's bytes and the whitespace before them as its source: a string still
    /// open, in either quote, as an open JSON string holding its text so far ([`open_string`]); a
    /// value whole before the separator that never came (`"Paris"`, `3`, `None`) as it would
    /// have been written ([`Self::whole_value`]); a number's digits, the start of a word or a
    /// list or a dict still open as the JSON they stand for so far ([`cut_word`], [`cut_json`]:
    /// as written where that is a prefix of a JSON value, a string holding the text where it is
    /// not); a key with no byte of its value yet as `"key": ` alone. Nothing is made up: no
    /// closing quote, bracket or brace, and every fragment keeps the arguments a strict prefix of
    /// a JSON value.
    fn cut_value(
        &mut self,
        key: &str,
        start: usize,
        scan: Extent,
        kind: Option<Kind>,
        out: &mut Events,
    ) {
        let raw = self.carried[start..].trim_start();
        let value = match scan.quote {
            _ if raw.is_empty() => String::new(),
            Some(quote) if scan.depth == 0 && raw.starts_with(quote) => {
                open_string(&raw[quote.len_utf8()..], quote, self.values)
            }
            None if scan.depth == 0 => {
                let text = raw.trim_end();
                self.whole_value(text, kind)
                    .unwrap_or_else(|| cut_word(text))
            }
            // A list or a dict still open: as written where its spelling is JSON's so far, else
            // a string holding it, since its JSON is not known until it closes.
            _ => cut_json(raw),
        };
        self.push_member(key, &value, out);
    }

    /// The JSON a value whole before its separator stands for, when its text is one the dialect
    /// reads whole: the text between a quoted value's quotes typed by `kind` under
    /// [`Values::Quoted`], a Python or JSON literal that reads back under every dialect; nothing
    /// for text that is no whole value yet.
    fn whole_value(&self, text: &str, kind: Option<Kind>) -> Option<String> {
        if self.values == Values::Quoted {
            if let Some(inner) = between_double_quotes(text) {
                return Some(json(inner, kind));
            }
        }
        python_literal(text).filter(|json| reads_back(json))
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

    /// What is held after the bare list, given back: the model's prose after its calls once text
    /// followed the list, whitespace included; whitespace nothing followed as the template's
    /// wrapping.
    fn release_after(&mut self, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        let text = Text::uncounted(std::mem::take(&mut self.carried));
        out.push(if self.prose_after {
            Event::Malformed {
                text,
                why: MalformedReason::TextAfterTheList,
            }
        } else {
            Event::Dropped {
                text,
                why: DropReason::Wrapper,
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
/// backslash and an `r` in a value are those two characters, not a carriage return, and a
/// backslash at the value's end is the value's last character, before the quote that ends it), typed by
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

/// A bare word or a number the end of the stream cut at the top of a value, as the JSON it
/// stands for so far: the start of one of Python's words is the start of the JSON word it is
/// written as (`Tr` is `tr`, `Non` is `nul`; the words are the same length), and anything else
/// is the JSON its text stands for as far as it is known ([`cut_json`]): a number's digits or the
/// start of a JSON word as written, other text as the opening of a string holding it.
fn cut_word(text: &str) -> String {
    for (word, json) in [("True", "true"), ("False", "false"), ("None", "null")] {
        if !text.is_empty() && word.starts_with(text) {
            return json[..text.len()].to_owned();
        }
    }
    cut_json(text)
}

/// A string the end of the stream cut open, as the open JSON string the dialect would have
/// written for its text so far: `inner`, the bytes after the opening `quote`, decoded as the
/// dialect reads them and escaped as JSON requires, after the opening `"` and with no closing
/// one. Under [`Values::Quoted`] nothing is decoded, so the text stands as written; under the
/// other dialects the string is closed as the model would have closed it and read whole, the
/// closing quote then taken off again, and when the cut left an escape unfinished (`\u00`) the
/// text from the last backslash that begins an escape is left out, since what it would have
/// stood for is not known, the escapes before it decoded as before; text the literal reader
/// cannot read even so stands as written.
fn open_string(inner: &str, quote: char, values: Values) -> String {
    if values != Values::Quoted {
        let decoded = [Some(inner.len()), last_escape_start(inner)]
            .into_iter()
            .flatten()
            .find_map(|end| python_literal(&format!("{quote}{}{quote}", &inner[..end])));
        if let Some(mut json) = decoded.filter(|json| json.starts_with('"') && json.len() > 1) {
            json.pop();
            return json;
        }
    }
    let mut json = String::with_capacity(inner.len() + 1);
    json.push('"');
    push_escaped(&mut json, inner);
    json
}

/// Where the last escape of `text` begins: the byte of its last backslash that is not itself the
/// character an earlier backslash escapes, if there is one.
fn last_escape_start(text: &str) -> Option<usize> {
    let mut start = None;
    let mut bytes = text.bytes().enumerate();
    while let Some((at, byte)) = bytes.next() {
        if byte == b'\\' {
            start = Some(at);
            bytes.next();
        }
    }
    start
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
    use serde_json::json as value;

    use super::*;
    use crate::testing::{bytes, content, tool};

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

    /// Under the quoted dialect a double-quoted value is the template's text as it stands: a
    /// backslash before the closing quote is the value's last character, not an escape, so the
    /// value closes at its quote, whole and at every cut, in the turn's list and in a list held
    /// after content alike; a single-quoted value is the model's Python, escapes and all, and the
    /// Python dialect keeps the escape in a double-quoted value too.
    #[test]
    fn a_quoted_value_ending_in_a_backslash_closes_at_its_quote() {
        let output = "[get_weather(city=\"dir C:\\\", unit=\"N/A\")]";
        let expected = vec![(
            0,
            "get_weather".to_string(),
            "{\"city\": \"dir C:\\\\\", \"unit\": \"N/A\"}".to_string(),
        )];
        assert_eq!(calls(&run_bare(&tools(), &[output])), expected);
        for at in 1..output.len() {
            let events = run_bare(&tools(), &[&output[..at], &output[at..]]);
            assert_eq!(calls(&events), expected, "cut at {at}");
            assert_eq!(malformed(&events), "", "cut at {at}");
        }
        let held = format!("Let me check.{output}");
        let events = run_bare(&tools(), &[&held]);
        assert_eq!(content(&events), "Let me check.");
        assert_eq!(calls(&events), expected);
        // A list the model writes bare is Python's: the quote inside its string is escaped, and
        // stays so in the turn's list and in a list held after content.
        let bare = "[get_weather(tags=[\"a\\\"b\", \"c\"], unit=\"N/A\")]";
        let expected = vec![(
            0,
            "get_weather".to_string(),
            "{\"tags\": [\"a\\\"b\", \"c\"], \"unit\": \"N/A\"}".to_string(),
        )];
        assert_eq!(calls(&run_bare(&tools(), &[bare])), expected);
        let held = format!("Let me check.{bare}");
        let events = run_bare(&tools(), &[&held]);
        assert_eq!(content(&events), "Let me check.");
        assert_eq!(calls(&events), expected);
        let events = run_bare(&tools(), &["[get_weather(city='it\\'s', unit=\"N/A\")]"]);
        assert_eq!(
            calls(&events),
            [(
                0,
                "get_weather".to_string(),
                "{\"city\": \"it's\", \"unit\": \"N/A\"}".to_string()
            )]
        );
        let events = run_with(
            Dialect::LFM2_5,
            &tools(),
            &["[get_weather(city=\"a\\\"b\", unit=\"N/A\")]"],
        );
        assert_eq!(
            calls(&events),
            [(
                0,
                "get_weather".to_string(),
                "{\"city\": \"a\\\"b\", \"unit\": \"N/A\"}".to_string()
            )]
        );
    }

    /// The quoted dialect's rule on every shape of backslash a value may hold, as the recorded
    /// benchmark sets write them: a double-quoted value is the template's bytes as they stand, so
    /// a backslash inside it is one of the value's characters, however many stand before the
    /// closing quote and whatever follows one, and the next `"` ends the value, save that a `"`
    /// right after a backslash is the model's own escaped quote unless the value's end follows
    /// it. Each shape whole, at every cut, in the turn's list and in a list held after content.
    #[test]
    fn a_quoted_values_backslashes_are_its_characters_wherever_they_stand() {
        let declared = Declared::of(&[tool(
            "cmd_controller_execute",
            value!({"type": "object", "properties": {
                "command": {"type": "string"},
                "unit": {"type": "string"},
            }}),
        )]);
        let shapes: &[(&str, &str)] = &[
            // one backslash before the closing quote, an argument after it and as the last
            (
                "[cmd_controller_execute(command=\"dir C:\\\", unit=\"N/A\")]",
                "{\"command\": \"dir C:\\\\\", \"unit\": \"N/A\"}",
            ),
            (
                "[cmd_controller_execute(command=\"dir C:\\\")]",
                "{\"command\": \"dir C:\\\\\"}",
            ),
            // two, and three
            (
                "[cmd_controller_execute(command=\"dir C:\\\\\", unit=\"N/A\")]",
                "{\"command\": \"dir C:\\\\\\\\\", \"unit\": \"N/A\"}",
            ),
            (
                "[cmd_controller_execute(command=\"dir C:\\\\\\\")]",
                "{\"command\": \"dir C:\\\\\\\\\\\\\"}",
            ),
            // a backslash with text after it, which no escape decodes (`\x` is not one here)
            (
                "[cmd_controller_execute(command=\"C:\\x\", unit=\"N/A\")]",
                "{\"command\": \"C:\\\\x\", \"unit\": \"N/A\"}",
            ),
            // a backslash before a comma inside the value, and before a single quote
            (
                "[cmd_controller_execute(command=\"a\\, b\\'c\", unit=\"N/A\")]",
                "{\"command\": \"a\\\\, b\\\\'c\", \"unit\": \"N/A\"}",
            ),
            // two calls, the first ending in a backslash
            (
                "[cmd_controller_execute(command=\"dir C:\\\"), cmd_controller_execute(unit=\"N/A\")]",
                "",
            ),
            // whitespace between the quote and the value's end
            (
                "[cmd_controller_execute(command=\"dir C:\\\" , unit=\"N/A\" )]",
                "{\"command\": \"dir C:\\\\\", \"unit\": \"N/A\"}",
            ),
            // a quote the model escaped Python's way, inside the value: the value's end does not
            // follow it, so the string goes on and the backslash is kept, as the family's own
            // readers keep the quote; once, twice, and right before the closing quote
            (
                "[cmd_controller_execute(command=\"5\\\" screen\", unit=\"N/A\")]",
                "{\"command\": \"5\\\\\\\" screen\", \"unit\": \"N/A\"}",
            ),
            (
                "[cmd_controller_execute(command=\"say \\\"hi\\\" now\", unit=\"N/A\")]",
                "{\"command\": \"say \\\\\\\"hi\\\\\\\" now\", \"unit\": \"N/A\"}",
            ),
            (
                "[cmd_controller_execute(command=\"say \\\"hi\\\"\", unit=\"N/A\")]",
                "{\"command\": \"say \\\\\\\"hi\\\\\\\"\", \"unit\": \"N/A\"}",
            ),
            (
                "[cmd_controller_execute(command=\"a\\\"b\", unit=\"N/A\")]",
                "{\"command\": \"a\\\\\\\"b\", \"unit\": \"N/A\"}",
            ),
        ];
        for (output, arguments) in shapes {
            let expected: Vec<(u32, String, String)> = if arguments.is_empty() {
                vec![
                    (
                        0,
                        "cmd_controller_execute".to_string(),
                        "{\"command\": \"dir C:\\\\\"}".to_string(),
                    ),
                    (
                        1,
                        "cmd_controller_execute".to_string(),
                        "{\"unit\": \"N/A\"}".to_string(),
                    ),
                ]
            } else {
                vec![(
                    0,
                    "cmd_controller_execute".to_string(),
                    arguments.to_string(),
                )]
            };
            let whole = run_bare(&declared, &[output]);
            assert_eq!(calls(&whole), expected, "{output}");
            assert_eq!(malformed(&whole), "", "{output}");
            assert_eq!(bytes(&whole), *output, "{output}");
            for at in 1..output.len() {
                let events = run_bare(&declared, &[&output[..at], &output[at..]]);
                assert_eq!(calls(&events), expected, "{output} cut at {at}");
                assert_eq!(malformed(&events), "", "{output} cut at {at}");
                assert_eq!(bytes(&events), *output, "{output} cut at {at}");
            }
            let held = format!("Listing the directory.{output}");
            let whole = run_bare(&declared, &[&held]);
            assert_eq!(content(&whole), "Listing the directory.", "{held}");
            assert_eq!(calls(&whole), expected, "{held}");
            assert_eq!(bytes(&whole), held, "{held}");
            for at in 1..held.len() {
                let events = run_bare(&declared, &[&held[..at], &held[at..]]);
                assert_eq!(
                    content(&events),
                    "Listing the directory.",
                    "{held} cut at {at}"
                );
                assert_eq!(calls(&events), expected, "{held} cut at {at}");
                assert_eq!(bytes(&events), held, "{held} cut at {at}");
            }
        }
        // A quote the model escaped right before the value's end is read as the template writes
        // it: the value ends there, with the backslash as its last character, and what follows
        // is no argument, so the turn's list ends open and says so with no arguments made up, and
        // a list held after content is content.
        let output = "[cmd_controller_execute(command=\"say \\\"hi\\\", ok\", unit=\"N/A\")]";
        let events = run_bare(&declared, &[output]);
        let read = calls(&events);
        assert_eq!(read.len(), 1);
        assert!(
            read[0]
                .2
                .starts_with("{\"command\": \"say \\\\\\\"hi\\\\\""),
            "{}",
            read[0].2
        );
        assert_ne!(malformed(&events), "");
        assert_eq!(bytes(&events), output);
        let held = format!("Listing the directory.{output}");
        let events = run_bare(&declared, &[&held]);
        assert_eq!(content(&events), held);
        assert!(calls(&events).is_empty());
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

    /// What a chat client reads as content once a call started: the text of every `Content` and
    /// `Malformed` event after the first `ToolCallStart`.
    fn shown_after_the_start(events: &[Event]) -> String {
        events
            .iter()
            .skip_while(|event| !matches!(event, Event::ToolCallStart { .. }))
            .filter_map(|event| match event {
                Event::Content(text) | Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The events of one region under `dialect` fed in `pieces` and cut by the stream's end.
    fn cut_with(dialect: Dialect, declared: &Declared, pieces: &[&str]) -> Vec<Event<'static>> {
        let mut assembler = Assembler::new(0, ids(), dialect);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, declared, &mut out);
        }
        assembler.finish(&mut out);
        out.drain()
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
        // The region's marker, the model's own end: the call ends with no bytes of its own and
        // no fragment, and its one word carries the value cut short.
        let events = Ending::Close.run(&["get_weather(city=\"Par"]);
        assert_eq!(bytes(&events), "get_weather(city=\"Par");
        assert!(
            matches!(
                &events[..],
                [
                    Event::ToolCallStart { index: 0, .. },
                    Event::Malformed { why: MalformedReason::UnterminatedRegion, .. },
                    Event::ToolCallEnd { index: 0, source },
                ] if source.text.is_empty()
            ),
            "{events:?}"
        );
        assert_eq!(said(&events), ["city=\"Par"]);
        // The stream's end: the value as far as it went is the call's own fragment, the open
        // string as written, and the one word of no bytes still says that the arguments are no
        // whole value.
        let events = Ending::Stream.run(&["get_weather(city=\"Par"]);
        assert_eq!(bytes(&events), "get_weather(city=\"Par");
        assert!(
            matches!(
                &events[..],
                [
                    Event::ToolCallStart { index: 0, .. },
                    Event::ToolCallArguments { json, source, .. },
                    Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                    Event::ToolCallEnd { index: 0, source: end },
                ] if json == r#"{"city": "Par"#
                    && source.text == "city=\"Par"
                    && text.text.is_empty()
                    && end.text.is_empty()
            ),
            "{events:?}"
        );
        assert_eq!(said(&events), [""]);
    }

    /// A call the end of the stream cut inside a member, after a key or between members, under
    /// the three dialects: the member leaves as the call's own fragment, its key and the JSON the
    /// assembler would have written for the value as far as that is known (an open string as an
    /// open JSON string, decoded as the dialect reads it; a value whole before its separator
    /// written whole; a number's digits, a word or a container still open as written; a key and
    /// no byte of its value as the key alone), a key or a separator cut short leaves as wrapping,
    /// nothing of the call reaches a chat client as content, and the one `Malformed` event of no
    /// bytes still says that the arguments are no whole value before the call's end; whole and
    /// one character at a time.
    #[test]
    fn a_member_the_stream_end_cuts_leaves_as_the_calls_own_fragment_and_nothing_as_content() {
        let cases: [(Dialect, &str, &str, &[&str]); 30] = [
            // JSON values: an open string, decoded and escaped as JSON.
            (
                Dialect::OLMO3,
                "get_weather(city=\"Par",
                r#"{"city": "Par"#,
                &[],
            ),
            (
                Dialect::OLMO3,
                "get_weather(city=\"Pa\\\"r",
                r#"{"city": "Pa\"r"#,
                &[],
            ),
            // An escape the cut left open is left out, what it stood for not being known, and
            // the escapes before it are still decoded.
            (
                Dialect::OLMO3,
                "get_weather(city=\"Pa\\",
                r#"{"city": "Pa"#,
                &[],
            ),
            (
                Dialect::OLMO3,
                "get_weather(city=\"caf\\u00e9 au lait\\u00",
                r#"{"city": "café au lait"#,
                &[],
            ),
            // A second member: a number's digits as written, after the first member whole.
            (
                Dialect::OLMO3,
                "get_weather(city=\"Paris\", days=1",
                r#"{"city": "Paris", "days": 1"#,
                &[],
            ),
            // A nested object cut inside a string: as written.
            (
                Dialect::OLMO3,
                "get_weather(city=\"Paris\", options={\"a\": \"b",
                r#"{"city": "Paris", "options": {"a": "b"#,
                &[],
            ),
            // The start of a literal word, and one whole before its separator; a word that is
            // no literal is a string's opening.
            (
                Dialect::OLMO3,
                "get_weather(detailed=tr",
                r#"{"detailed": tr"#,
                &[],
            ),
            (
                Dialect::OLMO3,
                "get_weather(city=hel",
                r#"{"city": "hel"#,
                &[],
            ),
            (
                Dialect::OLMO3,
                "get_weather(detailed=true",
                r#"{"detailed": true"#,
                &[],
            ),
            (
                Dialect::OLMO3,
                "get_weather(city=\"Paris\"",
                r#"{"city": "Paris""#,
                &[],
            ),
            // The key came and no byte of its value.
            (Dialect::OLMO3, "get_weather(city=", r#"{"city": "#, &[]),
            (Dialect::OLMO3, "get_weather(city= ", r#"{"city": "#, &[]),
            // A key cut short with the separator before it, and the separator alone: wrapping.
            (
                Dialect::OLMO3,
                "get_weather(city=\"Paris\", da",
                r#"{"city": "Paris""#,
                &[", da"],
            ),
            (
                Dialect::OLMO3,
                "get_weather(city=\"Paris\", ",
                r#"{"city": "Paris""#,
                &[", "],
            ),
            (
                Dialect::OLMO3,
                "get_weather(city=\"Paris\",",
                r#"{"city": "Paris""#,
                &[","],
            ),
            (Dialect::OLMO3, "get_weather(ci", "", &["ci"]),
            (Dialect::OLMO3, "get_weather(", "", &[]),
            // Python values: a single-quoted string opens as JSON's double-quoted one, its
            // escapes decoded; a literal whole before its separator is written whole; a dict
            // still open is as written.
            (
                Dialect::LFM2_5,
                "[get_weather(city='Par",
                r#"{"city": "Par"#,
                &[],
            ),
            (
                Dialect::LFM2_5,
                "[get_weather(city='it\\'s fi",
                r#"{"city": "it's fi"#,
                &[],
            ),
            (
                Dialect::LFM2_5,
                "[get_weather(city='Pa\"r",
                r#"{"city": "Pa\"r"#,
                &[],
            ),
            (
                Dialect::LFM2_5,
                "[get_weather(detailed=True",
                r#"{"detailed": true"#,
                &[],
            ),
            // The start of a Python word is the start of its JSON word.
            (
                Dialect::LFM2_5,
                "[get_weather(detailed=Tr",
                r#"{"detailed": tr"#,
                &[],
            ),
            (
                Dialect::LFM2_5,
                "[get_weather(city=None",
                r#"{"city": null"#,
                &[],
            ),
            (
                Dialect::LFM2_5,
                "[get_weather(city=Non",
                r#"{"city": nul"#,
                &[],
            ),
            // A dict still open, whose JSON is not known until it closes: a string holding its
            // text, so that the arguments stay a strict prefix of a JSON value.
            (
                Dialect::LFM2_5,
                "[get_weather(city='Paris', options={'a': 'b",
                r#"{"city": "Paris", "options": "{'a': 'b"#,
                &[],
            ),
            // Quoted values: the text between the quotes as written, nothing decoded; a value
            // whole before its separator typed by the tool.
            (
                Dialect::LLAMA4,
                "[get_weather(city=\"Par",
                r#"{"city": "Par"#,
                &[],
            ),
            (
                Dialect::LLAMA4,
                "[get_weather(city=\"Paris\", days=\"3",
                r#"{"city": "Paris", "days": "3"#,
                &[],
            ),
            (
                Dialect::LLAMA4,
                "[get_weather(city=\"Paris\", days=\"3\"",
                r#"{"city": "Paris", "days": 3"#,
                &[],
            ),
            (
                Dialect::LLAMA4,
                "[get_weather(city=\"Paris\", options=\"{'a': 'b",
                r#"{"city": "Paris", "options": "{'a': 'b"#,
                &[],
            ),
            (Dialect::LLAMA4, "[get_weather(city=", r#"{"city": "#, &[]),
        ];
        for (dialect, call, expected, wrapping) in cases {
            let characters = by_the_character(call);
            let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
            for (how, pieces) in [("whole", vec![call]), ("by character", characters)] {
                let events = cut_with(dialect, &tools(), &pieces);
                let place = format!("{dialect:?} {call:?} {how}");
                assert_eq!(bytes(&events), call, "{place}");
                let calls = calls(&events);
                assert_eq!(calls.len(), 1, "{place}: {events:?}");
                assert_eq!(calls[0].2, expected, "{place}: {events:?}");
                // The contract of every fragment: the arguments so far are a strict prefix of a
                // JSON value, one `serde_json` reads whole or stops reading only at its end.
                let strict_prefix = match serde_json::from_str::<serde_json::Value>(&calls[0].2) {
                    Ok(_) => true,
                    Err(error) => error.is_eof(),
                };
                assert!(
                    strict_prefix,
                    "{place}: no JSON value begins with {:?}",
                    calls[0].2
                );
                assert!(
                    shown_after_the_start(&events).is_empty(),
                    "{place}: nothing of the call is content: {events:?}"
                );
                assert_eq!(words(&events), [""], "{place}: {events:?}");
                assert!(
                    matches!(
                        &events[events.len() - 2..],
                        [
                            Event::Malformed { why: MalformedReason::UnterminatedRegion, .. },
                            Event::ToolCallEnd { source, .. }
                        ] if source.text.is_empty()
                    ),
                    "{place}: {events:?}"
                );
                let dropped_after_the_start: Vec<&str> = events
                    .iter()
                    .skip_while(|event| !matches!(event, Event::ToolCallStart { .. }))
                    .filter_map(|event| match event {
                        Event::Dropped { text, .. } => Some(&*text.text),
                        _ => None,
                    })
                    .collect();
                assert_eq!(dropped_after_the_start, wrapping, "{place}: {events:?}");
            }
        }
        // The bare list, a second call cut inside its value: the first whole, the list's
        // wrapping between them, the second call's fragment.
        let events = run_bare(
            &tools(),
            &["[get_weather(city=\"Paris\"), get_weather(city=\"Lon"],
        );
        assert_eq!(
            calls(&events),
            [
                (
                    0,
                    "get_weather".to_string(),
                    r#"{"city": "Paris"}"#.to_string()
                ),
                (1, "get_weather".to_string(), r#"{"city": "Lon"#.to_string()),
            ]
        );
        assert!(shown_after_the_start(&events).is_empty(), "{events:?}");
        assert_eq!(words(&events), [""]);
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
    fn text_after_the_bare_list_is_the_models_prose_or_wrapping_and_a_cut_list_stays_cut() {
        // Once text follows the list, every byte after the `]` is the model's prose, the
        // whitespace before its first word and after its last included.
        let text = "[get_weather(city=\"Paris\")]\n\nDone.\n";
        let events = run_bare(&tools(), &[text]);
        assert_eq!(bytes(&events), text);
        assert_eq!(content(&events), "");
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(malformed(&events), "\n\nDone.\n");
        assert!(events.iter().all(|event| !matches!(
            event,
            Event::Malformed { why, .. } if *why != MalformedReason::TextAfterTheList
        )));
        assert_eq!(dropped(&events), ["[", "]"]);
        // Whitespace alone after the list is the template's wrapping, however it arrives.
        for pieces in [
            &["[get_weather(city=\"Paris\")]\n\n"][..],
            &["[get_weather(city=\"Paris\")]", "\n", "\n"],
        ] {
            let events = run_bare(&tools(), pieces);
            assert_eq!(calls(&events).len(), 1);
            assert_eq!(malformed(&events), "");
            assert_eq!(dropped(&events), ["[", "]", "\n\n"]);
        }
        // Cut inside a value: the call ends with nothing of its own, and the value, held until
        // its extent is known, is the call's own fragment as far as it went, the open string as
        // written; the one word of no bytes says that the arguments are no whole value.
        let events = run_bare(&tools(), &["[get_weather(city=\"Par"]);
        assert_eq!(content(&events), "");
        assert_eq!(
            calls(&events),
            [(0, "get_weather".to_string(), r#"{"city": "Par"#.to_string())]
        );
        assert_eq!(words(&events), [""]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::UnterminatedRegion }
                if text.text.is_empty()
        )));
    }

    #[test]
    fn prose_after_a_list_written_mid_sentence_keeps_its_whitespace_whole_and_in_pieces() {
        // The model explains which call it would make, the list inside the sentence: the words
        // after the list keep the spaces between them, one `Malformed` event per word with the
        // whitespace before it, whole, at every two-way cut and one character at a time.
        let text = "For example, [get_weather(city=\"Honolulu\")] would give you the answer.";
        let prose = " would give you the answer.";
        let whole = run_bare(&tools(), &[text]);
        assert_eq!(bytes(&whole), text);
        assert_eq!(content(&whole), "For example, ");
        assert_eq!(
            calls(&whole),
            [(
                0,
                "get_weather".to_string(),
                r#"{"city": "Honolulu"}"#.to_string()
            )]
        );
        assert_eq!(malformed(&whole), prose);
        assert_eq!(
            words(&whole),
            [" would", " give", " you", " the", " answer."]
        );
        assert_eq!(dropped(&whole), ["[", "]"]);
        for cut in 1..text.len() {
            let events = run_bare(&tools(), &[&text[..cut], &text[cut..]]);
            assert_eq!(bytes(&events), text, "cut at {cut}");
            assert_eq!(calls(&events), calls(&whole), "cut at {cut}");
            assert_eq!(content(&events), "For example, ", "cut at {cut}");
            assert_eq!(malformed(&events), prose, "cut at {cut}");
            assert_eq!(dropped(&events), ["[", "]"], "cut at {cut}");
        }
        let events = run_bare(
            &tools(),
            &by_the_character(text)
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );
        assert_eq!(bytes(&events), text);
        assert_eq!(calls(&events), calls(&whole));
        assert_eq!(content(&events), "For example, ");
        assert_eq!(malformed(&events), prose);
        assert_eq!(dropped(&events), ["[", "]"]);
        // A trailing run of whitespace after the prose is the prose's too, and a word cut by a
        // delta boundary leaves in two pieces, the same text joined.
        let text = "[get_weather(city=\"Paris\")]\n\nDone, see above.\n";
        let whole = run_bare(&tools(), &[text]);
        assert_eq!(malformed(&whole), "\n\nDone, see above.\n");
        assert_eq!(words(&whole), ["\n\nDone,", " see", " above.", "\n"]);
        let events = run_bare(
            &tools(),
            &[
                "[get_weather(city=\"Paris\")]\n\nDo",
                "ne, see ",
                "above.\n",
            ],
        );
        assert_eq!(malformed(&events), malformed(&whole));
        assert_eq!(words(&events), ["\n\nDo", "ne,", " see", " above.", "\n"]);
        assert_eq!(dropped(&events), ["[", "]"]);
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
                            // At the close the one word carries what was held; at the stream's
                            // end the held bytes are the call's own fragment or its wrapping,
                            // the word has no bytes, and nothing of the call is shown.
                            match ending {
                                Ending::Close => {
                                    assert_eq!(said(&events), [*held], "{place}: what was held");
                                }
                                Ending::Stream => {
                                    assert_eq!(said(&events), [""], "{place}: a word of no bytes");
                                    assert!(
                                        shown_after_the_start(&events).is_empty(),
                                        "{place}: nothing of the call is shown: {events:?}"
                                    );
                                }
                            }
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
