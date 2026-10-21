//! Call shapes as data: the patterns a template writes around a call's parts, and a matcher that
//! runs them while the output streams.
//!
//! A table under [`formats`] spells a family's markers as terminals and leaves what lies between
//! them to the assembler its call syntax names. A model whose template publishes its own reader, a
//! Hugging Face response template, says more of the call, as regular expressions: what follows the
//! call's marker and names the tool (`<function=(?P<name>[^>\n]+)>`), a parameter's tag with its
//! key and its value (`<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>`), what comes
//! before the call's closing marker (`</function>\s*`). This module holds such expressions as data
//! and runs them over text as it arrives, so that a call syntax can be given as data rather than
//! written as one more assembler.
//!
//! - A [`Pattern`] is linear: [`Atom`]s matched one after another, each a literal, a run of one
//!   [`Class`] of characters (greedy or lazy, bounded or not, captured as the tool's name, a key or
//!   a value when its [`Slot`] says so), `\b`, or an optional literal. The patterns public
//!   response templates write inside a call lower to one; a compiler refuses the rest by name.
//! - An [`End`] is one way a value ends: text the match takes, then text it only looks ahead at.
//! - A [`Class`] is a set of characters as ranges. Whoever lowers a template resolves `\w`, `\s`
//!   and `\d` into ranges with the template's own dialect, so the crate holds no character
//!   tables.
//! - A [`Matcher`] runs a pattern, with its ends, over text as it arrives. Its answer is the one
//!   Python's `regex` module gives for the whole text (`match` under `DOTALL`, the dialect response
//!   templates are written in), and it gives it as soon as no continuation of the text can change
//!   it.
//! - A [`Suffix`] runs a pattern from every place of a text as it arrives, and says where the
//!   earliest match that ends at the text's end starts: where the text before a closing marker
//!   (`</function>\s*` before `</tool_call>`) begins, when the marker comes.
//! - [`Pattern::check`] refuses a pattern that a stream could decide only at the end of the
//!   output.
//! - An [`Interner`] keeps one copy of each distinct value for the life of the process, so a table
//!   built at run time can hold its shapes as `&'static` data and stay `Copy`.
//!
//! A [`CallShape`] puts the patterns together into a call: the header after the region's
//! terminal, which may capture the tool's name, the way the arguments are written
//! ([`Arguments`]: one JSON value, an object that names the call, or tagged parameters, each a
//! tag with its key and its value and the ends a value may have), and the trailer before the
//! region's closing terminal. A table holds one as [`CallSyntax::Shaped`], [`intern`]ed, and the
//! shape's assembler turns a call written so into its events, giving back to the engine the
//! bytes of a header that did not match. A value's text reads by the template's [`ValueParser`]
//! and is typed by the table's [`ValueRead`]er, [`Inferred`] unless the table names another.
//!
//! A [`Layout`] says how the model's chat template writes a turn where the patterns leave the text
//! open: which whitespace a `\s*` stands for, which attributes a skip passes, what the text outside
//! every field is. The structural tags of a shaped call are written from it.
//!
//! [`formats`]: crate::formats
//! [`CallSyntax::Shaped`]: crate::CallSyntax::Shaped

mod call;
mod class;
mod intern;
mod layout;
mod matcher;
mod params;
mod read;

use std::fmt;

pub(crate) use call::{Assembler, Header};
pub use class::Class;
pub use intern::Interner;
pub use layout::{
    CallLayout, Layout, LayoutError, Message, Nested, Segment, Spellings, Words, Written,
};
use matcher::reads_through;
pub use matcher::{Match, Matcher, Suffix, Verdict, HOLD_CAP};
pub(crate) use read::Values;
pub use read::{Duplicates, Inferred, Schema, ValueParser, ValueRead};

/// A linear pattern: its atoms, matched one after another from the start of the text it is run
/// over.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Pattern(pub Box<[Atom]>);

/// One step of a [`Pattern`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Atom {
    /// These characters, in order.
    Lit(Box<str>),
    /// A run of characters of `class`: `min` at least, `max` at most (no bound when `None`), as
    /// many as the rest of the pattern allows when `greedy` and as few when not, captured as
    /// `capture` says: `\w+`, `[^"]+`, `\S+?`, `\s*`, `\n?`, `.*?`, `[^>]*?`.
    Run {
        class: Class,
        min: u16,
        max: Option<u16>,
        greedy: bool,
        capture: Option<Slot>,
    },
    /// `\b`: a word character on one side and none on the other, by the word class the pattern is
    /// run with.
    /// Before the text's start and after its end there is no word character.
    Boundary,
    /// `(?:LIT)?`: these characters, or nothing when the rest of the pattern does not match after
    /// them.
    Maybe(Box<str>),
}

/// What a captured run is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    /// The tool's name: `(?P<name>…)`.
    Name,
    /// A parameter's key: `(?P<key>…)`.
    Key,
    /// A parameter's value: `(?P<value>…)`.
    Value,
}

/// One way a value ends, as a parameter's tag spells it: `consume` is part of the match
/// (`\s*</parameter>`), and `ahead` must follow it but is not (a value that also ends where the
/// next parameter's tag begins, `(?=<parameter=)`). Either may be empty.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct End {
    pub consume: Pattern,
    pub ahead: Pattern,
}

/// Why a stream cannot run a pattern. An atom is named by its place in its pattern.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PatternError {
    /// A literal with no characters, which the lowering should have left out.
    #[error("atom {atom}: a literal with no characters")]
    EmptyLiteral { atom: usize },
    /// A run whose least number of characters is above its most.
    #[error("atom {atom}: a run of at least {min} and at most {max} characters")]
    Bounds { atom: usize, min: u16, max: u16 },
    /// A greedy run of every character with no bound. No text after it ends the run, so the
    /// stream would hold everything that follows until the output ends.
    #[error("atom {atom}: a greedy run of any character with no bound")]
    Undecidable { atom: usize },
    /// A second run captured as a slot one run already captures.
    #[error("atom {atom}: a second run captured as {slot:?}")]
    CapturedTwice { atom: usize, slot: Slot },
    /// An end with nothing to match and nothing to look ahead at: every value would end where it
    /// starts.
    #[error("an end with nothing to match or to look ahead at")]
    EmptyEnd,
}

impl Pattern {
    /// Whether a stream can run the pattern: every literal has characters, every run's bounds are
    /// in order, no slot is captured twice, and every run decides before the output ends. A run
    /// that cannot is a greedy one of every character with no bound (`.*`, `[\s\S]*`): it takes
    /// all the text there is and gives it back only when nothing comes after it. A greedy run of a
    /// narrower class ends at the first character outside it (`\S+` at a space), and a lazy run
    /// ends at the first place the rest of the pattern matches, so neither needs the end of the
    /// output to decide.
    pub fn check(&self) -> Result<(), PatternError> {
        let mut captured: Vec<Slot> = Vec::new();
        for (atom, step) in self.0.iter().enumerate() {
            match step {
                Atom::Lit(text) | Atom::Maybe(text) if text.is_empty() => {
                    return Err(PatternError::EmptyLiteral { atom });
                }
                Atom::Run {
                    min,
                    max: Some(max),
                    ..
                } if min > max => {
                    return Err(PatternError::Bounds {
                        atom,
                        min: *min,
                        max: *max,
                    });
                }
                Atom::Run {
                    class,
                    max: None,
                    greedy: true,
                    ..
                } if class.is_any() => return Err(PatternError::Undecidable { atom }),
                _ => {}
            }
            if let Atom::Run {
                capture: Some(slot),
                ..
            } = step
            {
                if captured.contains(slot) {
                    return Err(PatternError::CapturedTwice { atom, slot: *slot });
                }
                captured.push(*slot);
            }
        }
        Ok(())
    }
}

impl End {
    /// Whether a stream can run this end: one of its parts holds something, and each part passes
    /// [`Pattern::check`].
    pub fn check(&self) -> Result<(), PatternError> {
        if self.consume.0.is_empty() && self.ahead.0.is_empty() {
            return Err(PatternError::EmptyEnd);
        }
        self.consume.check()?;
        self.ahead.check()
    }
}

/// A call given as data: what the template writes after the region's terminal and before the
/// arguments, how it writes the arguments, and what it writes after them before the region's
/// closing terminal. A table holds one as [`CallSyntax::Shaped`], [`intern`]ed so that the call
/// syntax stays `Copy`; [`CallShape::check`] says whether a stream can run it.
///
/// [`CallSyntax::Shaped`]: crate::CallSyntax::Shaped
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CallShape {
    /// What follows the region's terminal, matched from its end: `\s*<function=(?P<name>\w+)>`.
    /// It may capture the tool's name. A terminal whose header does not match opens no call, and
    /// is the text of the state it came in. An empty header opens the call at the terminal.
    pub header: Pattern,
    /// Where the tool's name comes from.
    pub name: NameSource,
    /// How the arguments are written.
    pub arguments: Arguments,
    /// What comes after the arguments, right before the region's closing terminal:
    /// `</function>\s*`, or the `}` of an object the header opened. A closing terminal the
    /// trailer does not end at closes nothing, and is the call's text. An empty trailer ends
    /// wherever the terminal comes. The stream's stop, where the table says it closes the call,
    /// closes tagged parameters wherever the trailer stands.
    pub trailer: Pattern,
    /// The characters `\b` reads as a word's: the template dialect's `\w`, which whoever lowered
    /// the shape resolved.
    pub word: Class,
    /// The most bytes a header or a trailer holds undecided; one longer is no match. [`HOLD_CAP`]
    /// unless a shape says otherwise.
    pub hold_cap: u32,
}

/// Where a shaped call's name comes from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NameSource {
    /// The header's run captured as [`Slot::Name`]: `<function=(?P<name>\w+)>`.
    Header,
    /// The arguments' own object, which names the call: [`Arguments::Envelope`].
    Body,
    /// The template's text: a transform that writes the name itself.
    Literal(Box<str>),
}

/// How a shaped call's arguments are written.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Arguments {
    /// One JSON value, the arguments themselves, the call named by the header or the template
    /// (`content: json`). They stream as the model's bytes, but for a string member of the top
    /// level the table's [`ValueRead`]er holds to type it.
    Json(Dialect),
    /// One JSON object that names the call and holds its arguments (`"function": "{content}"`):
    /// the name at `name`, the arguments at `arguments`, read by the JSON syntax's own reader and
    /// streamed as the model's bytes; the table's reader is not asked about their members.
    Envelope {
        dialect: Dialect,
        name: Path,
        arguments: Path,
    },
    /// Tagged parameters (`content: xml-inline`): each a `tag` that begins with a literal, runs
    /// through the parameter's key and value, captured as [`Slot::Key`] and [`Slot::Value`], and
    /// ends with the first of `ends` that matches (`\s*</parameter>`, or a lookahead at the next
    /// parameter's tag). A parameter counts once its end matched, as a regular expression's
    /// `finditer` finds it; text between parameters is skipped, and each value is written as JSON
    /// when its parameter ends, its text read by `values` and typed by the table's
    /// [`ValueRead`]er.
    Params {
        tag: Pattern,
        ends: Box<[End]>,
        values: ValueParser,
    },
}

/// The JSON a template writes: plain JSON, or a dialect of it, keys left bare and strings between
/// the template's own delimiters, as transformers' `json` parser reads with `unquoted_keys` and
/// `string_delims`. The default is plain JSON.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dialect {
    /// Keys written without quotes.
    pub bare_keys: bool,
    /// The delimiters around a string, each an opening and a closing one, written in place of
    /// JSON's quotes; the text between them is the string as it is.
    pub string_delims: Box<[(Box<str>, Box<str>)]>,
}

impl Dialect {
    /// Whether this is plain JSON.
    pub fn is_plain(&self) -> bool {
        !self.bare_keys && self.string_delims.is_empty()
    }
}

/// Where a member stands in a JSON object, key by key from the top: `["name"]`, or `["call",
/// "args"]` for a dotted `{content.call.args}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Path(pub Box<[Box<str>]>);

impl Path {
    /// The path through `keys`, from the top.
    pub fn new<'k>(keys: impl IntoIterator<Item = &'k str>) -> Self {
        Self(keys.into_iter().map(Box::from).collect())
    }
}

/// The keys the envelope reader takes a call's name from, the first of them in the object.
const NAME_KEYS: [&str; 2] = ["name", "tool_name"];
/// The keys it takes the arguments from.
const ARGUMENTS_KEYS: [&str; 3] = ["arguments", "parameters", "args"];

/// The part of a call shape a pattern stands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Header,
    Trailer,
    /// A parameter's tag.
    Tag,
    /// A value's end, by its place in the list.
    End(usize),
}

impl fmt::Display for Part {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header => formatter.write_str("header"),
            Self::Trailer => formatter.write_str("trailer"),
            Self::Tag => formatter.write_str("parameter's tag"),
            Self::End(index) => write!(formatter, "value's end {index}"),
        }
    }
}

/// Why a stream cannot run a call shape.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ShapeError {
    /// A pattern a stream cannot run.
    #[error("the {part}: {error}")]
    Pattern { part: Part, error: PatternError },
    /// A run captured as something the part has no place for: a key or a value in the header,
    /// anything in the trailer.
    #[error("the {part} captures a {slot:?}, which it has no place for")]
    Capture { part: Part, slot: Slot },
    /// The name comes from the header, which captures none.
    #[error("the name comes from the header, which captures none")]
    NoName,
    /// The name comes from the arguments, which are no object that names the call.
    #[error("the name comes from the arguments, which are no envelope")]
    NoEnvelope,
    /// The arguments are an object that names the call, and the name comes from elsewhere too.
    #[error("the envelope names the call, and the name comes from elsewhere too")]
    NamedTwice,
    /// JSON in a dialect other than plain JSON, which no reader here takes yet.
    #[error("JSON in a dialect other than plain JSON")]
    Dialect,
    /// An envelope path the envelope reader does not take: it reads the name from `name` or
    /// `tool_name` and the arguments from `arguments`, `parameters` or `args`, each at the top.
    #[error("an envelope path the envelope reader does not take: {0:?}")]
    EnvelopePath(Path),
    /// A parameter's tag that does not begin with a literal: the reader finds parameters by the
    /// text they begin with.
    #[error("a parameter's tag that does not begin with a literal")]
    NoTrigger,
    /// A parameter's tag that does not capture `slot` once.
    #[error("a parameter's tag that does not capture its {0:?}")]
    Uncaptured(Slot),
    /// Parameters with no way for a value to end.
    #[error("parameters with no end for a value")]
    NoEnds,
    /// A value's end that reads past what it matches (a lookahead, or anything but a literal at
    /// its end: a `\b`, a run, an optional literal) in a call with a trailer: the trailer may turn
    /// out to begin there, where the call's text ends, so the end could not be decided as the
    /// bytes arrive.
    #[error("value's end {0} reads past itself, where the trailer may begin")]
    ReadsPastTheText(usize),
    /// A value's end whose last literal the trailer could read, from inside it or from before it,
    /// in a call with a trailer: the call's text ends where the trailer begins, so the end could be
    /// the trailer's text, and the parameter none.
    #[error("the trailer could read value's end {0}, where the call's text ends")]
    TrailerReadsTheEnd(usize),
}

impl CallShape {
    /// Whether a stream can run the shape: the header and the trailer pass [`Pattern::check`]; the
    /// header captures only the name and the trailer nothing; the name comes from where the shape
    /// says, the header capturing it, or an envelope naming the call and nothing else doing so;
    /// the JSON is plain; an envelope's paths are ones its reader takes; and parameters are found
    /// by the literal their tag begins with, the tag captures the key and the value once each,
    /// there is an end for a value, each end passes [`End::check`] and captures nothing, and with a
    /// trailer every end is decided by a last literal the trailer cannot read.
    pub fn check(&self) -> Result<(), ShapeError> {
        for (part, pattern) in [(Part::Header, &self.header), (Part::Trailer, &self.trailer)] {
            pattern
                .check()
                .map_err(|error| ShapeError::Pattern { part, error })?;
            let allowed = |slot: Slot| part == Part::Header && slot == Slot::Name;
            if let Some(slot) = pattern.captures().find(|&slot| !allowed(slot)) {
                return Err(ShapeError::Capture { part, slot });
            }
        }
        let header_names = self.header.captures().any(|slot| slot == Slot::Name);
        match (&self.name, &self.arguments) {
            (NameSource::Header, _) if !header_names => return Err(ShapeError::NoName),
            (NameSource::Body, Arguments::Envelope { .. }) => {}
            (NameSource::Body, _) => return Err(ShapeError::NoEnvelope),
            (_, Arguments::Envelope { .. }) => return Err(ShapeError::NamedTwice),
            _ => {}
        }
        match &self.arguments {
            Arguments::Json(dialect) | Arguments::Envelope { dialect, .. }
                if !dialect.is_plain() =>
            {
                Err(ShapeError::Dialect)
            }
            Arguments::Envelope {
                name, arguments, ..
            } => {
                let top = |path: &Path, keys: &[&str]| match &*path.0 {
                    [key] => keys.contains(&key.as_ref()),
                    _ => false,
                };
                if !top(name, &NAME_KEYS) {
                    return Err(ShapeError::EnvelopePath(name.clone()));
                }
                if !top(arguments, &ARGUMENTS_KEYS) {
                    return Err(ShapeError::EnvelopePath(arguments.clone()));
                }
                Ok(())
            }
            Arguments::Json(_) => Ok(()),
            Arguments::Params { tag, ends, .. } => self.check_params(tag, ends),
        }
    }

    fn check_params(&self, tag: &Pattern, ends: &[End]) -> Result<(), ShapeError> {
        tag.check().map_err(|error| ShapeError::Pattern {
            part: Part::Tag,
            error,
        })?;
        if !matches!(tag.0.first(), Some(Atom::Lit(_))) {
            return Err(ShapeError::NoTrigger);
        }
        if let Some(slot) = tag.captures().find(|&slot| slot == Slot::Name) {
            return Err(ShapeError::Capture {
                part: Part::Tag,
                slot,
            });
        }
        for slot in [Slot::Key, Slot::Value] {
            if !tag.captures().any(|captured| captured == slot) {
                return Err(ShapeError::Uncaptured(slot));
            }
        }
        if ends.is_empty() {
            return Err(ShapeError::NoEnds);
        }
        for (index, end) in ends.iter().enumerate() {
            let part = Part::End(index);
            end.check()
                .map_err(|error| ShapeError::Pattern { part, error })?;
            let captured = end.consume.captures().chain(end.ahead.captures()).next();
            if let Some(slot) = captured {
                return Err(ShapeError::Capture { part, slot });
            }
            if !self.trailer.0.is_empty() {
                self.check_end_before_trailer(index, end)?;
            }
        }
        Ok(())
    }

    /// With a trailer, the call's text ends where the trailer begins, which only the closing
    /// terminal shows, and a parameter is written once its end matched: so an end must be decided
    /// by its own last character, the end of a literal, with nothing looked ahead at, and the
    /// trailer must not be able to read that literal, from inside it or from before it, or a
    /// parameter written could be text the trailer takes.
    fn check_end_before_trailer(&self, index: usize, end: &End) -> Result<(), ShapeError> {
        let last = match end.consume.0.last() {
            Some(Atom::Lit(last)) if end.ahead.0.is_empty() => last,
            _ => return Err(ShapeError::ReadsPastTheText(index)),
        };
        let (trailer, word) = (&self.trailer, &self.word);
        let inside = last.char_indices().skip(1).any(|(at, _)| {
            let before = last[..at].chars().next_back();
            reads_through(trailer, word, before, &last[at..], false)
        });
        // Begun before it, the trailer reads the literal after a character of any kind: none, and
        // a word's, for a `\b`.
        let a_word = ['a', 'A', '0', '_'].into_iter().find(|&c| word.contains(c));
        let whole = std::iter::once(None)
            .chain(a_word.map(Some))
            .any(|before| reads_through(trailer, word, before, last, true));
        if inside || whole {
            return Err(ShapeError::TrailerReadsTheEnd(index));
        }
        Ok(())
    }

    /// The literals a call of this shape writes as its own tags inside its region, which free text
    /// before a call may not write and a client that shows a call as text strips: the one a
    /// parameter's tag begins with, and each literal of a value's ends longer than one character
    /// and not whitespace alone. A header's and a trailer's literals are the template's text
    /// around a name or after the arguments (`name="`, `}`), too common to keep out of free text,
    /// and arguments written as JSON have no tags of their own.
    pub fn tag_texts<'s>(&'s self) -> Vec<&'s str> {
        let Arguments::Params { tag, ends, .. } = &self.arguments else {
            return Vec::new();
        };
        let mut texts: Vec<&'s str> = Vec::new();
        let mut keep = |text: &'s str| {
            if !texts.contains(&text) {
                texts.push(text);
            }
        };
        if let Some(Atom::Lit(trigger)) = tag.0.first() {
            keep(trigger);
        }
        for end in ends {
            for atom in end.consume.0.iter().chain(end.ahead.0.iter()) {
                if let Atom::Lit(lit) = atom {
                    if lit.chars().nth(1).is_some() && !lit.chars().all(char::is_whitespace) {
                        keep(lit);
                    }
                }
            }
        }
        texts
    }

    /// The cap on what a header or a trailer holds undecided, in bytes.
    pub(crate) fn cap(&self) -> usize {
        usize::try_from(self.hold_cap).unwrap_or(usize::MAX)
    }
}

impl Pattern {
    /// The slots the pattern's runs capture, in order.
    fn captures(&self) -> impl Iterator<Item = Slot> + '_ {
        self.0.iter().filter_map(|atom| match atom {
            Atom::Run {
                capture: Some(slot),
                ..
            } => Some(*slot),
            _ => None,
        })
    }
}

/// The process's one copy of `shape`, as [`CallSyntax::Shaped`] holds it: a shape built at run
/// time, compiled from a model's template, becomes `'static` data, kept for the life of the
/// process (see [`Interner`]).
///
/// [`CallSyntax::Shaped`]: crate::CallSyntax::Shaped
pub fn intern(shape: CallShape) -> &'static CallShape {
    static SHAPES: Interner<CallShape> = Interner::new();
    SHAPES.intern(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(class: Class, min: u16, max: Option<u16>, greedy: bool) -> Atom {
        Atom::Run {
            class,
            min,
            max,
            greedy,
            capture: None,
        }
    }

    fn captured(atom: Atom, slot: Slot) -> Atom {
        match atom {
            Atom::Run {
                class,
                min,
                max,
                greedy,
                ..
            } => Atom::Run {
                class,
                min,
                max,
                greedy,
                capture: Some(slot),
            },
            other => other,
        }
    }

    fn pattern(atoms: impl IntoIterator<Item = Atom>) -> Pattern {
        Pattern(atoms.into_iter().collect())
    }

    #[test]
    fn a_greedy_run_of_any_character_with_no_bound_is_refused() {
        let any = Class::any();
        let word = Class::new([('a', 'z')], false);
        assert_eq!(
            pattern([Atom::Lit("<a>".into()), run(any.clone(), 0, None, true)]).check(),
            Err(PatternError::Undecidable { atom: 1 })
        );
        let spelled = Class::new([('\0', '\u{D7FF}'), ('\u{E000}', char::MAX)], false);
        assert_eq!(
            pattern([run(spelled, 1, None, true)]).check(),
            Err(PatternError::Undecidable { atom: 0 }),
            "`[\\s\\S]+` is the same run"
        );
        for decidable in [
            run(any.clone(), 0, None, false),
            run(any.clone(), 0, Some(64), true),
            run(Class::new([('>', '>')], true), 0, None, true),
            run(word, 1, None, true),
        ] {
            assert_eq!(
                pattern([decidable.clone(), Atom::Lit(">".into())]).check(),
                Ok(()),
                "{decidable:?}"
            );
        }
    }

    #[test]
    fn literals_bounds_and_captures_are_checked_by_place() {
        let word = Class::new([('a', 'z')], false);
        assert_eq!(
            pattern([Atom::Boundary, Atom::Maybe("".into())]).check(),
            Err(PatternError::EmptyLiteral { atom: 1 })
        );
        assert_eq!(
            pattern([run(word.clone(), 3, Some(2), true)]).check(),
            Err(PatternError::Bounds {
                atom: 0,
                min: 3,
                max: 2
            })
        );
        assert_eq!(
            pattern([
                captured(run(word.clone(), 1, None, true), Slot::Name),
                Atom::Lit("=".into()),
                captured(run(word.clone(), 1, None, true), Slot::Name),
            ])
            .check(),
            Err(PatternError::CapturedTwice {
                atom: 2,
                slot: Slot::Name
            })
        );
        assert_eq!(
            pattern([
                captured(run(word.clone(), 1, None, true), Slot::Key),
                Atom::Lit(">".into()),
                captured(run(Class::any(), 0, None, false), Slot::Value),
            ])
            .check(),
            Ok(())
        );
    }

    #[test]
    fn an_end_needs_something_to_match_or_to_look_ahead_at() {
        assert_eq!(End::default().check(), Err(PatternError::EmptyEnd));
        let close = pattern([Atom::Lit("</a>".into())]);
        assert_eq!(
            End {
                consume: close.clone(),
                ahead: Pattern::default()
            }
            .check(),
            Ok(())
        );
        assert_eq!(
            End {
                consume: Pattern::default(),
                ahead: close
            }
            .check(),
            Ok(())
        );
        assert_eq!(
            End {
                consume: pattern([run(Class::any(), 0, None, true)]),
                ahead: Pattern::default()
            }
            .check(),
            Err(PatternError::Undecidable { atom: 0 })
        );
    }

    /// A shape on abstract tags: `<n=NAME>` after the terminal, the arguments as one JSON value.
    fn named() -> CallShape {
        let word = Class::new([('a', 'z')], false);
        CallShape {
            header: pattern([
                Atom::Lit("<n=".into()),
                captured(run(word.clone(), 1, None, true), Slot::Name),
                Atom::Lit(">".into()),
            ]),
            name: NameSource::Header,
            arguments: Arguments::Json(Dialect::default()),
            trailer: Pattern::default(),
            word,
            hold_cap: 8,
        }
    }

    fn envelope() -> Arguments {
        Arguments::Envelope {
            dialect: Dialect::default(),
            name: Path::new(["name"]),
            arguments: Path::new(["arguments"]),
        }
    }

    #[test]
    fn a_shape_names_its_call_from_one_place() {
        assert_eq!(named().check(), Ok(()));
        let literal = CallShape {
            name: NameSource::Literal("f".into()),
            ..named()
        };
        assert_eq!(
            literal.check(),
            Ok(()),
            "a name the template writes; the header's capture is not read"
        );
        let unnamed = CallShape {
            header: Pattern::default(),
            ..named()
        };
        assert_eq!(unnamed.check(), Err(ShapeError::NoName));
        let from_the_body = CallShape {
            name: NameSource::Body,
            ..named()
        };
        assert_eq!(from_the_body.check(), Err(ShapeError::NoEnvelope));
        let enveloped = CallShape {
            header: Pattern::default(),
            name: NameSource::Body,
            arguments: envelope(),
            ..named()
        };
        assert_eq!(enveloped.check(), Ok(()));
        let twice = CallShape {
            arguments: envelope(),
            ..named()
        };
        assert_eq!(twice.check(), Err(ShapeError::NamedTwice));
    }

    #[test]
    fn a_header_captures_only_the_name_and_a_trailer_nothing() {
        let word = Class::new([('a', 'z')], false);
        let keyed = CallShape {
            header: pattern([captured(run(word.clone(), 1, None, true), Slot::Key)]),
            name: NameSource::Literal("f".into()),
            ..named()
        };
        assert_eq!(
            keyed.check(),
            Err(ShapeError::Capture {
                part: Part::Header,
                slot: Slot::Key
            })
        );
        let trailing = CallShape {
            trailer: pattern([captured(run(word, 1, None, true), Slot::Name)]),
            ..named()
        };
        assert_eq!(
            trailing.check(),
            Err(ShapeError::Capture {
                part: Part::Trailer,
                slot: Slot::Name
            })
        );
        let undecidable = CallShape {
            trailer: pattern([run(Class::any(), 0, None, true)]),
            ..named()
        };
        assert_eq!(
            undecidable.check(),
            Err(ShapeError::Pattern {
                part: Part::Trailer,
                error: PatternError::Undecidable { atom: 0 }
            })
        );
        assert_eq!(
            undecidable.check().map_err(|error| error.to_string()),
            Err("the trailer: atom 0: a greedy run of any character with no bound".to_string())
        );
    }

    #[test]
    fn json_is_plain_and_an_envelope_is_read_where_its_reader_reads() {
        let bare = CallShape {
            arguments: Arguments::Json(Dialect {
                bare_keys: true,
                string_delims: Box::default(),
            }),
            ..named()
        };
        assert_eq!(bare.check(), Err(ShapeError::Dialect));
        let enveloped = |name: &[&str], arguments: &[&str]| CallShape {
            header: Pattern::default(),
            name: NameSource::Body,
            arguments: Arguments::Envelope {
                dialect: Dialect::default(),
                name: Path::new(name.iter().copied()),
                arguments: Path::new(arguments.iter().copied()),
            },
            ..named()
        };
        assert_eq!(enveloped(&["tool_name"], &["parameters"]).check(), Ok(()));
        assert_eq!(enveloped(&["name"], &["args"]).check(), Ok(()));
        assert_eq!(
            enveloped(&["function", "name"], &["arguments"]).check(),
            Err(ShapeError::EnvelopePath(Path::new(["function", "name"])))
        );
        assert_eq!(
            enveloped(&["name"], &["input"]).check(),
            Err(ShapeError::EnvelopePath(Path::new(["input"])))
        );
    }

    #[test]
    fn an_interned_shape_is_one_copy() {
        let first = intern(named());
        let again = intern(named());
        assert!(std::ptr::eq(first, again));
        assert_eq!(*first, named());
    }
}
