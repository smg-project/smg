//! Why a template has no table: every construct the compile does not lower, each named by a
//! stable code.

use std::{error::Error, fmt};

/// A construct of a response template that the compile does not lower into a symphony table,
/// named by a stable code. Logs, metrics and the pinned outcome lists use [`Construct::code`];
/// a new lowering removes a code's refusals, never renames it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Construct {
    /// Inline flags that change what a pattern matches (`(?i)`, `(?m)`, ...).
    InlineFlags,
    /// A backreference (`\1`, `(?P=name)`).
    Backreference,
    /// A lookbehind (`(?<=...)`).
    LookBehind,
    /// A negative lookahead or lookbehind (`(?!...)`, `(?<!...)`).
    NegativeLookAround,
    /// An open that is a lookahead as a whole (`(?=[\s\S]*?</think>)`): it matches before any
    /// text of its own, which a table reads only by guessing the region from the start.
    ZeroWidthOpen,
    /// Any other lookahead: one in a delimiter, before a tagged value, or with more after it in
    /// the way a value ends.
    LookAhead,
    /// `^` or `\A` (the `(?:^|<think>)` open of a region the output may start in).
    StartOfText,
    /// `$`, `\Z` or `\z` (the `|$` alternative of a close, or a value that ends where the call's
    /// text ends).
    EndOfText,
    /// `\B`.
    NotWordBoundary,
    /// A repeat of anything other than one character or an optional literal: a quantified
    /// group, nested or not.
    NestedRepeat,
    /// A possessive repeat (`*+`, `++`, `?+`).
    Possessive,
    /// A counted repeat with a bound above 65,535.
    LongRepeat,
    /// A Unicode property or a POSIX class.
    Property,
    /// A construct the `regex` module reads that no table reads: an atomic group, a branch reset
    /// group, a conditional, fuzzy matching, a group call, `\K`, `\G`, `\X`, `\R`, `\m`, `\M`, a
    /// verb.
    OtherRegex,
    /// An alternation the lowering cannot spell as several markers: more than 32 alternatives, or
    /// one inside a captured group.
    Alternation,
    /// A captured group that the transform reads, or a tag pattern's key or value, that is not
    /// one run of a class.
    Capture,
    /// A start anchor pattern that is not a finite set of literals.
    AnchorNotFinite,
    /// A delimiter with no literal text to be its marker: an open whose first piece after
    /// whitespace, or a close whose last piece before whitespace, is not a literal.
    NoLiteral,
    /// A text field's open or close with more than optional whitespace beside its marker (a
    /// filler such as `[^<]*`, a required run, a capture).
    TextFieldDelimiter,
    /// Whitespace a delimiter takes next to a state that keeps its whitespace: a field with
    /// `strip: false` or a state whose text is not trimmed.
    UntrimmedSlack,
    /// Whitespace a delimiter takes beside its marker, where another marker that moves the state
    /// the whitespace is read in may begin: after a close's marker, its greedy run would take that
    /// marker's first character; before a marker, the delimiter's match would start in the
    /// whitespace, ahead of that marker.
    SlackBeforeMarker,
    /// An alternative whose marker begins the marker of a later alternative: the pattern takes
    /// the shorter one first, a table the longer.
    AlternativeOrder,
    /// The field with no open is the thought or the calls; only content, or a field the message
    /// has no place for, can be the text outside every other field.
    ImplicitField,
    /// Two fields that are the same field of a message (`thinking` and `reasoning_content`).
    SharedField,
    /// A text field (the thought, the content) whose content is not `text`.
    TextFieldContent,
    /// A transform of a text field.
    TextFieldTransform,
    /// A text field that repeats (with or without `join`).
    RepeatedTextField,
    /// A default value for the thought, the content or the calls.
    Default,
    /// Tool calls that do not repeat: one region holds every call.
    NonRepeatingCalls,
    /// Tool calls given as a list (`transform_each`).
    TransformEach,
    /// A call transform other than `{"function": {"name": ..., "arguments": "{content}"}}` or an
    /// object that names the call.
    CallTransform,
    /// A transform placeholder with a dotted path (`{content.name}`).
    DottedPath,
    /// Tool calls whose content is `text`, `int`, `float` or `bool`: a call needs a name and
    /// arguments.
    CallContent,
    /// A call named by its open with its arguments as JSON. Not lowered yet.
    JsonArguments,
    /// A call that is one JSON object holding its name and arguments. Not lowered yet: the JSON
    /// reader of such an object writes its members as the model wrote them, where transformers
    /// types a string member by the tool's schema.
    JsonEnvelope,
    /// JSON written in a dialect (`unquoted_keys`, `string_delims`).
    JsonDialect,
    /// Tagged parameters (`xml-inline`) in a call that the parameters name themselves
    /// (`"function": "{content}"`): the call's name would be a parameter's value.
    TaggedParameters,
    /// Arguments as `key: value` lines (`kv-lines`).
    KeyValueLines,
    /// `merge_duplicates`.
    MergeDuplicates,
    /// The transform's name is no group the open captures once.
    HeaderName,
    /// A run in a call's header or trailer that a stream could decide only at the end of the
    /// output (a greedy run of any character with no bound).
    UnboundedHeaderRun,
    /// Two fields whose opens begin with the same marker.
    SharedOpenMarker,
    /// A marker that holds, or runs into, a marker that moves the state it is read in: the table
    /// would read the longer one where the template reads the other.
    MarkerInsideMarker,
    /// The assembled table does not pass the table's own check.
    Table,
    /// A tag pattern of tagged parameters that does not lower into one tag a stream runs: none
    /// that this crate's reader runs, a tag that does not begin with literal text, a key or a
    /// value that the tag does not capture exactly once, alternatives that differ before the
    /// value, a value with nothing after it, or a greedy run of any character with no bound.
    TagPattern,
    /// In a call with text before its closing marker (a trailer), a value's end that reads past
    /// itself (a lookahead, or a run, a `\b` or an optional literal at its end), or whose last
    /// literal the trailer could read: the call's text ends where the trailer begins, which only
    /// the closing marker shows, so such an end could not be decided as the text arrives.
    EndBeforeTrailer,
    /// A value parser that is itself structured (`xml-inline` or `kv-lines`): a tagged value
    /// reads as one value.
    NestedValueParser,
    /// A call's open whose alternatives differ after their markers, or a close whose alternatives
    /// differ before theirs: a call reads with one header and one trailer.
    CallAlternatives,
}

impl Construct {
    /// The stable code: the variant's name.
    pub fn code(self) -> &'static str {
        match self {
            Self::InlineFlags => "InlineFlags",
            Self::Backreference => "Backreference",
            Self::LookBehind => "LookBehind",
            Self::NegativeLookAround => "NegativeLookAround",
            Self::ZeroWidthOpen => "ZeroWidthOpen",
            Self::LookAhead => "LookAhead",
            Self::StartOfText => "StartOfText",
            Self::EndOfText => "EndOfText",
            Self::NotWordBoundary => "NotWordBoundary",
            Self::NestedRepeat => "NestedRepeat",
            Self::Possessive => "Possessive",
            Self::LongRepeat => "LongRepeat",
            Self::Property => "Property",
            Self::OtherRegex => "OtherRegex",
            Self::Alternation => "Alternation",
            Self::Capture => "Capture",
            Self::AnchorNotFinite => "AnchorNotFinite",
            Self::NoLiteral => "NoLiteral",
            Self::TextFieldDelimiter => "TextFieldDelimiter",
            Self::UntrimmedSlack => "UntrimmedSlack",
            Self::SlackBeforeMarker => "SlackBeforeMarker",
            Self::AlternativeOrder => "AlternativeOrder",
            Self::ImplicitField => "ImplicitField",
            Self::SharedField => "SharedField",
            Self::TextFieldContent => "TextFieldContent",
            Self::TextFieldTransform => "TextFieldTransform",
            Self::RepeatedTextField => "RepeatedTextField",
            Self::Default => "Default",
            Self::NonRepeatingCalls => "NonRepeatingCalls",
            Self::TransformEach => "TransformEach",
            Self::CallTransform => "CallTransform",
            Self::DottedPath => "DottedPath",
            Self::CallContent => "CallContent",
            Self::JsonArguments => "JsonArguments",
            Self::JsonEnvelope => "JsonEnvelope",
            Self::JsonDialect => "JsonDialect",
            Self::TaggedParameters => "TaggedParameters",
            Self::KeyValueLines => "KeyValueLines",
            Self::MergeDuplicates => "MergeDuplicates",
            Self::HeaderName => "HeaderName",
            Self::UnboundedHeaderRun => "UnboundedHeaderRun",
            Self::SharedOpenMarker => "SharedOpenMarker",
            Self::MarkerInsideMarker => "MarkerInsideMarker",
            Self::Table => "Table",
            Self::TagPattern => "TagPattern",
            Self::EndBeforeTrailer => "EndBeforeTrailer",
            Self::NestedValueParser => "NestedValueParser",
            Self::CallAlternatives => "CallAlternatives",
        }
    }
}

impl fmt::Display for Construct {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// One construct the compile did not lower: where it stands in the template (`response_template`
/// or `Field 'name'`), its code, and what it is there.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{scope}: {construct}: {detail}")]
pub struct Refusal {
    pub scope: String,
    pub construct: Construct,
    pub detail: String,
}

/// Every construct a template uses that the compile does not lower, in the order the compile
/// met them. Displayed as the first; [`Refusals::refusals`] lists them all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusals(pub(super) Vec<Refusal>);

impl Refusals {
    /// Every refusal, in the order the compile met them.
    pub fn refusals(&self) -> &[Refusal] {
        &self.0
    }

    /// The distinct codes, sorted.
    pub fn constructs(&self) -> Vec<Construct> {
        let mut codes: Vec<Construct> = self.0.iter().map(|refusal| refusal.construct).collect();
        codes.sort_unstable();
        codes.dedup();
        codes
    }
}

impl fmt::Display for Refusals {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.first() {
            Some(first) => first.fmt(formatter),
            None => formatter.write_str("no refusal"),
        }
    }
}

impl Error for Refusals {}

/// A refusal before its scope is known: the construct and what it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Refused(pub(super) Construct, pub(super) String);

impl Refused {
    pub(super) fn new(construct: Construct, detail: impl Into<String>) -> Self {
        Self(construct, detail.into())
    }

    pub(super) fn scoped(self, scope: &str) -> Refusal {
        Refusal {
            scope: scope.to_owned(),
            construct: self.0,
            detail: self.1,
        }
    }
}
