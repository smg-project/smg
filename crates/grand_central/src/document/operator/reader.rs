//! Read the source's bytes and name a refusal by the field it came from.

use std::{cell::Cell, fmt, path::Path};

use serde::de::{self, DeserializeSeed, Deserializer, EnumAccess, MapAccess, SeqAccess, Visitor};
use serde_json::{map::Entry, Map, Number, Value};
use serde_path_to_error::Segment;

use crate::document::spec::PolicySpec;

/// The largest document a source hands over, in bytes: 128 KiB. What sets
/// the bound is what an apply may hold beside the document, 64 MiB: the
/// budget of one apply in a controller whose container is a few hundred
/// mebibytes. By run a YAML document of this size holds under 56 MiB.
/// An apply runs one at a time, and a dry run one at a time
/// beside it when the host takes its turns. A fleet's document is far under the bound: 143 names
/// with a binding each write 11,523 bytes.
pub const MAX_DOCUMENT_BYTES: u64 = 128 << 10;

/// A document's size, where it is over the bound.
pub(crate) fn over_bound(document: &[u8]) -> Option<u64> {
    let len = u64::try_from(document.len()).unwrap_or(u64::MAX);
    (len > MAX_DOCUMENT_BYTES).then_some(len)
}

/// How many bytes a YAML document may build for each byte of the size bound
/// ([`MAX_DOCUMENT_BYTES`]), as the values it is read into. A value weighs
/// what it holds: its own size and its text, and a key its own size, its
/// text and its place in the map. A document of the size bound that writes
/// its values out builds under the bound: the densest form, a flow map of
/// one-letter keys with no values (`{a,b}`), builds 65 bytes a byte. What an
/// alias multiplies has no bound in the bytes.
///
/// The bound is the largest document's whatever a document's own size: a
/// small file that shares one profile among its names by alias reads as it
/// does written out, and no document reads as more than one of the size
/// bound may.
const BUILT_A_BYTE: usize = 80;

/// How many bytes of text a YAML document may read as for each byte of the
/// size bound, its keys among them. A document that writes its text out
/// reads as at most one and a half for each of its own (an escape of two
/// bytes writes a character of three). Past the read a document is its text
/// and its values, so under this bound an alias makes no document read as
/// more text than twice the size bound.
const TEXT_A_BYTE: usize = 2;

/// Why a document passes a bound its bytes are under: said in each refusal.
const MULTIPLIED: &str = "an alias reads as its anchor's value at each use";

/// What a value holds wherever it stands: its own size.
const NODE: usize = size_of::<Value>();

/// What a key holds beside its text: its own size, its hash and its place
/// in the map's index.
const KEY: usize = size_of::<String>() + 4 * size_of::<usize>();

/// A document's form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Format {
    /// A document in YAML.
    Yaml,
    /// A document in JSON.
    Json,
}

impl Format {
    /// A file's form by its name: a name with the extension `json`, in any
    /// letter case, is JSON, any other name YAML. A file named only `.json`
    /// has no extension and is YAML. The one rule for every host that starts
    /// from a file.
    pub fn of(path: &Path) -> Self {
        let json = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"));
        if json {
            Self::Json
        } else {
            Self::Yaml
        }
    }

    /// The name of this form.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Yaml => "yaml",
            Self::Json => "json",
        }
    }

    /// The form a contract's `format` field names, as [`Format::as_str`]
    /// writes one; none for any other name.
    pub fn named(name: &str) -> Option<Self> {
        [Self::Yaml, Self::Json]
            .into_iter()
            .find(|format| format.as_str() == name)
    }

    /// The document these bytes read as. A document of either form is read
    /// once, into the values it reads as ([`read_yaml`], [`read_json`]),
    /// and the document is built from those values alone: one typed read
    /// for both forms. So a YAML document reads as JSON does: text that
    /// looks like a number or a boolean is quoted where a field takes text,
    /// and a whole number past 64 bits reads as a fraction. A number reads
    /// one way in both forms and in every build: a whole number of 64 bits
    /// as itself, and any other as the nearest double. A narrower field
    /// takes it by a cast. So a number that rounds past the largest single
    /// reads as infinite where a single stands, and the spec's check
    /// refuses it by its field. A field the reader refuses is named by its
    /// path, and a YAML document's refusal says what its writer does about
    /// a plain word ([`cured`]).
    ///
    /// Three numbers read apart, by the two forms' own rules. A zero
    /// written `-0` is the whole number zero in YAML and a fraction in
    /// JSON, which a whole-number field refuses. A number past a double is
    /// out of range in JSON and text in YAML, which a number's field
    /// refuses. And a YAML whole number may be written in hex, octal or
    /// binary.
    ///
    /// The size bound is not checked here: a source's bytes are read by
    /// [`Format::read`], which checks it first.
    pub(super) fn parse(self, document: &[u8]) -> Result<PolicySpec, Refused> {
        match self {
            Self::Yaml => {
                placed(read_yaml(document).map_err(Refused::whole)?).map_err(|refused| Refused {
                    message: cured(refused.message, refused.item),
                    ..refused
                })
            }
            Self::Json => placed(read_json(document).map_err(Refused::whole)?),
        }
    }
}

/// The typed read of a document's values: a refusal carries the path of the
/// field the read refused.
fn placed(values: Value) -> Result<PolicySpec, Refused> {
    serde_path_to_error::deserialize(values).map_err(|refused| {
        let item = matches!(refused.path().iter().next_back(), Some(Segment::Seq { .. }));
        let path = refused.path().to_string();
        let message = refused.into_inner().to_string();
        // The document itself has no name.
        let path = if path == "." { String::new() } else { path };
        Refused {
            path: wanted(path, &message),
            message,
            item,
        }
    })
}

/// `message` with what a writer of YAML does about it, where the typed read
/// refused a plain word. A number or a boolean where the field takes text
/// is that text in quotes. A null is a key with nothing under it, or a list
/// item with nothing in it where the refusal's place is one (`item`): the
/// key or the item is taken out, or given its text, its list or its map.
fn cured(message: String, item: bool) -> String {
    let cure = message
        .strip_prefix("invalid type: ")
        .and_then(|rest| rest.split_once(", expected "))
        .and_then(|(found, expected)| {
            let word = found.split([' ', '`']).next().unwrap_or(found);
            let text = expected == "a string";
            let holds =
                ["a sequence", "a map"].contains(&expected) || expected.starts_with("struct ");
            match word {
                "integer" | "floating" | "boolean" if text => {
                    Some("text that reads as a number, a boolean or a null is written in quotes")
                }
                "null" if text && item => Some(
                    "a list item with nothing in it is a null: take the item out, or write its text in quotes",
                ),
                "null" if text => Some(
                    "a key with nothing under it is a null: take the key out, or write its text in quotes",
                ),
                "null" if holds && item => Some(
                    "a list item with nothing in it is a null: take the item out, or write [] or {}",
                ),
                "null" if holds => Some(
                    "a key with nothing under it is a null: take the key out, or write [] or {}",
                ),
                _ => None,
            }
        });
    match cure {
        Some(cure) => format!("{message}: {cure}"),
        None => message,
    }
}

/// `path` ended by the key a typed read found missing. The map that wants
/// the key raises that refusal with no key in hand, so the read places it at
/// the map, the document itself for a key of the root, and the reader's
/// message names the key. A key a map does not know is placed at itself.
fn wanted(mut path: String, message: &str) -> String {
    let missing = message
        .strip_prefix("missing field `")
        .and_then(|rest| rest.split_once('`'));
    if let Some((key, _)) = missing {
        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(key);
    }
    path
}

/// Why a reader refused a document: the path of the field, where the
/// refusal is about one, the reader's message, and whether the place is a
/// list's item, which has no key.
pub(super) struct Refused {
    pub(super) path: String,
    pub(super) message: String,
    item: bool,
}

impl Refused {
    /// A refusal of the document as a whole.
    fn whole(message: impl fmt::Display) -> Self {
        Self {
            path: String::new(),
            message: message.to_string(),
            item: false,
        }
    }
}

/// Reads a YAML document once, into the values it reads as, held under two
/// bounds.
///
/// A document's bytes do not bound what it reads as: an alias reads as its
/// anchor's value at each use, so one list of a thousand values under a
/// thousand aliases is a million values, and one long text under a thousand
/// aliases is a thousand copies of it. Each value is weighed as it is held,
/// by the bytes it holds, and the read ends where the weight passes
/// [`BUILT_A_BYTE`] for each byte of the size bound, or its text alone
/// [`TEXT_A_BYTE`]. What is held is what was weighed (a list and a map keep
/// no room they do not fill), and a document is built from it alone: nothing
/// a document builds is past the bounds either.
///
/// The values are JSON's: a value or a key under a local tag, a number JSON
/// does not write and a key written twice are each refused by name. A
/// global tag is read as YAML reads it, by the parser and ahead of this
/// read: one of the four that type a scalar types it or refuses its text,
/// any other leaves a scalar its text, and none changes a list, a map or a
/// key.
///
/// What an apply holds beside the file is found by run and bounded by no
/// form: a search of the heaviest shapes finds at most 367 to 406 bytes for
/// each byte of the size bound, by the run and the build, and holds every
/// file under 448, 56 MiB (the measurement
/// `what_a_file_at_the_bound_holds`, run by name; on Linux every run of the
/// tests holds its heaviest file to the budget). Two things no bound of the
/// read sees are the larger part. The parser lists a file's nodes whole
/// before the first value is read, three nodes to two bytes at the most and
/// about a hundred bytes a node. And a document's types outweigh the values
/// they are read from: a binding of nothing is read as one value and held as
/// a type six values wide.
/// The heaviest file found writes bindings of nothing to the size bound
/// beside one profile of a thousand scorers that its aliases share among a
/// hundred names.
///
/// The parser has a limit of its own, a hundred aliases followed for each
/// node a file writes: a small nest of aliases meets it before either bound
/// and is refused in the parser's words. It bounds a nest too, in its own
/// words: a list reads 128 deep, and a map that holds a key 127 deep, the
/// JSON form's depth, because a key's read holds one level.
pub(super) fn read_yaml(document: &[u8]) -> Result<Value, String> {
    let left = Left::whole();
    Weighed(&left)
        .deserialize(serde_yaml::Deserializer::from_slice(document))
        .map_err(|error| worded(error.to_string()))
}

/// A refusal of the YAML read, with a key that is a list or a map refused
/// as it was where the parser's string entry read a key: it wants a key of
/// text. A key is asked of the enum entry ([`Key`]), which words such a key
/// as wanting a tag, and a writer who adds one is refused again.
///
/// The rewording is by the parser's whole phrase, for each of the two
/// kinds: a document that writes that phrase as text of its own has it
/// reworded in a refusal that quotes the text.
fn worded(refusal: String) -> String {
    ["sequence", "map"].iter().fold(refusal, |refusal, kind| {
        refusal.replace(
            &format!("invalid type: {kind}, expected a YAML tag starting with '!'"),
            &format!("invalid type: {kind}, expected a key of text"),
        )
    })
}

/// Reads a JSON document once, into the values it reads as, by the visitor
/// a YAML document's values are read by ([`read_yaml`]). A key written
/// twice is refused by name, a field's as a map's, at the place where its
/// map ends. A JSON document has no alias, so one of the size bound is
/// under both bounds of the read: its densest form, a list of one-digit
/// numbers, builds one value for each two bytes. The library has a limit
/// of its own: a document nested 128 deep is refused in the library's
/// words.
///
/// The document is not built from its bytes. That read asks the JSON
/// library for a single-precision number where a single stands, and the
/// library answers by its build. With its exact reading it refuses a number
/// past single precision and rounds a long number straight to a single,
/// and without it does neither. A value is the nearest double in every
/// build that asks for the exact reading, and this crate's manifest asks
/// for it.
fn read_json(document: &[u8]) -> Result<Value, serde_json::Error> {
    let left = Left::whole();
    let mut reader = serde_json::Deserializer::from_slice(document);
    let values = Weighed(&left).deserialize(&mut reader)?;
    reader.end()?;
    Ok(values)
}

/// What a read has left of a document's two bounds.
struct Left {
    /// Of the bytes it may build.
    built: Cell<usize>,
    /// Of the text it may read as.
    text: Cell<usize>,
}

impl Left {
    /// The two bounds, whole: those of a document of the size bound.
    fn whole() -> Self {
        let bytes = usize::try_from(MAX_DOCUMENT_BYTES).unwrap_or(usize::MAX);
        Self {
            // The document itself is a value, an empty one too.
            built: Cell::new(bytes.saturating_mul(BUILT_A_BYTE).saturating_add(NODE)),
            text: Cell::new(bytes.saturating_mul(TEXT_A_BYTE)),
        }
    }

    /// `bytes` more built.
    fn hold<E: de::Error>(&self, bytes: usize) -> Result<(), E> {
        let Some(left) = self.built.get().checked_sub(bytes) else {
            return Err(E::custom(format_args!(
                "the document builds more than {BUILT_A_BYTE} bytes for each byte of the size bound ({MAX_DOCUMENT_BYTES}): {MULTIPLIED}"
            )));
        };
        self.built.set(left);
        Ok(())
    }

    /// `bytes` more of text, read and built.
    fn read<E: de::Error>(&self, bytes: usize) -> Result<(), E> {
        let Some(left) = self.text.get().checked_sub(bytes) else {
            return Err(E::custom(format_args!(
                "the document reads as more than {TEXT_A_BYTE} bytes of text for each byte of the size bound ({MAX_DOCUMENT_BYTES}): {MULTIPLIED}"
            )));
        };
        self.text.set(left);
        self.hold(bytes)
    }
}

/// One node of a document, read into its value against what the bounds have
/// left.
struct Weighed<'a>(&'a Left);

impl Weighed<'_> {
    /// One value more, of `text` bytes of text.
    fn weigh<E: de::Error>(&self, text: usize) -> Result<(), E> {
        self.0.hold(NODE)?;
        self.0.read(text)
    }
}

impl<'de> DeserializeSeed<'de> for Weighed<'_> {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Weighed<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a value JSON writes")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        self.weigh(0)?;
        Ok(Value::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Value, E> {
        self.weigh(0)?;
        Ok(Value::from(value))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        self.weigh(0)?;
        Ok(Value::from(value))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        self.weigh(0)?;
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom(format_args!("the number {value} is none JSON writes")))
    }

    /// A whole number past 64 bits reads as a fraction, as the JSON form
    /// reads it.
    fn visit_i128<E: de::Error>(self, value: i128) -> Result<Value, E> {
        self.visit_f64(value as f64)
    }

    fn visit_u128<E: de::Error>(self, value: u128) -> Result<Value, E> {
        self.visit_f64(value as f64)
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Value, E> {
        self.weigh(text.len())?;
        Ok(Value::String(text.to_owned()))
    }

    /// A value under a local tag, refused by the tag's name: JSON has no
    /// form for one. A global tag does not come here: the parser reads it.
    fn visit_enum<A: EnumAccess<'de>>(self, tagged: A) -> Result<Value, A::Error> {
        let (tag, _) = tagged.variant::<String>()?;
        Err(local_tag(&tag))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        self.weigh(0)?;
        Ok(Value::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        self.visit_unit()
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        self.weigh(0)?;
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(Weighed(self.0))? {
            items.push(item);
        }
        items.shrink_to_fit();
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        self.weigh(0)?;
        let mut read = Vec::new();
        while let Some(key) = map.next_key_seed(Key(self.0))? {
            read.push((key, map.next_value_seed(Weighed(self.0))?));
        }
        let mut fields = Map::with_capacity(read.len());
        for (key, value) in read {
            match fields.entry(key) {
                Entry::Vacant(place) => {
                    place.insert(value);
                }
                Entry::Occupied(first) => {
                    return Err(de::Error::custom(format_args!(
                        "the key `{}` is written twice",
                        first.key()
                    )));
                }
            }
        }
        Ok(Value::Object(fields))
    }
}

/// The refusal of a local tag by its name, a value's and a key's in the same
/// words.
fn local_tag<E: de::Error>(tag: &str) -> E {
    E::custom(format_args!(
        "the tag `!{}` is none JSON writes: take the tag out",
        tag.trim_start_matches('!')
    ))
}

/// A map's key, read as its text: a key that looks like a number is its
/// digits. A key under a local tag is refused by the tag's name, as a value
/// under one is.
///
/// A parser's string entry hands a key's text and no tag. So a key is asked
/// of the enum entry, the one entry that hands a key's tag over, and what
/// that entry hands is read by [`KeyVariant`].
struct Key<'a>(&'a Left);

impl<'de> DeserializeSeed<'de> for Key<'_> {
    type Value = String;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<String, D::Error> {
        deserializer.deserialize_enum("", &[], self)
    }
}

impl<'de> Visitor<'de> for Key<'_> {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a key of text")
    }

    fn visit_enum<A: EnumAccess<'de>>(self, key: A) -> Result<String, A::Error> {
        key.variant_seed(KeyVariant(self.0)).map(|(text, _)| text)
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<String, E> {
        self.0.hold(KEY)?;
        self.0.read(text.len())?;
        Ok(text.to_owned())
    }
}

/// A key as a parser's enum entry hands it over, asked for a newtype struct
/// to tell the two things it may be. A key with no local tag is the
/// document's own reader, still in front of the key: the string entry reads
/// its text ([`Key`]). A key under a local tag is the tag's name, and that
/// is the refusal.
struct KeyVariant<'a>(&'a Left);

impl<'de> DeserializeSeed<'de> for KeyVariant<'_> {
    type Value = String;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<String, D::Error> {
        deserializer.deserialize_newtype_struct("", self)
    }
}

impl<'de> Visitor<'de> for KeyVariant<'_> {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a key of text")
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<String, D::Error> {
        deserializer.deserialize_str(Key(self.0))
    }

    fn visit_str<E: de::Error>(self, tag: &str) -> Result<String, E> {
        Err(local_tag(tag))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/document/operator/reader/tests.rs"]
mod tests;
