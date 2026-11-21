//! Read the source's bytes and name a refusal by the field it came from.

#[cfg(any(test, feature = "controller"))]
use std::path::Path;
use std::{cell::Cell, fmt};

use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{map::Entry, Map, Number, Value};

use crate::document::spec::PolicySpec;

/// The largest document a source hands over, in bytes: 128 KiB. What sets
/// the bound is what an apply may hold beside the document, 64 MiB: the
/// budget of one apply in a controller whose container is a few hundred
/// mebibytes. By run a YAML document of this size holds under 56 MiB.
/// An apply runs one at a time, and a dry run one at a time
/// beside it when the host takes its turns. A fleet's document is far under the bound: 143 names
/// with a binding each write 11,523 bytes.
pub const MAX_DOCUMENT_BYTES: u64 = 128 << 10;

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
pub enum Format {
    /// A document in YAML.
    Yaml,
    /// A document in JSON.
    Json,
}

impl Format {
    /// A file's form by its name: `.json` is JSON, any other name YAML.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn of(path: &Path) -> Self {
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

    /// The document these bytes read as. A YAML document is read once, into
    /// the values it reads as, held under a bound ([`read_yaml`]), and the
    /// document is built from those values alone. So a YAML document reads
    /// as JSON does: text that looks like a number or a boolean is quoted
    /// where a field takes text. A field the reader refuses is named by its
    /// path.
    pub(super) fn parse(self, document: &[u8]) -> Result<PolicySpec, Refused> {
        match self {
            Self::Yaml => placed(read_yaml(document).map_err(Refused::whole)?),
            // A JSON document that reads is read once, by the typed read
            // alone. A refused one is read again as values and placed as a
            // YAML document's are, for the field of its refusal.
            Self::Json => serde_json::from_slice(document).map_err(|refused| {
                match serde_json::from_slice(document).map(placed) {
                    Ok(Err(at_field)) => at_field,
                    // No values, or values that read (a key written twice
                    // keeps its last): the refusal is the typed read's own.
                    Ok(Ok(_)) | Err(_) => Refused::whole(refused),
                }
            }),
        }
    }
}

/// The typed read of a document's values: a refusal carries the path of the
/// field the read refused.
fn placed(values: Value) -> Result<PolicySpec, Refused> {
    serde_path_to_error::deserialize(values).map_err(|refused| {
        let path = refused.path().to_string();
        let message = refused.into_inner().to_string();
        // The document itself has no name.
        let path = if path == "." { String::new() } else { path };
        Refused {
            path: wanted(path, &message),
            message,
        }
    })
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
/// refusal is about one, and the reader's message.
pub(super) struct Refused {
    pub(super) path: String,
    pub(super) message: String,
}

impl Refused {
    /// A refusal of the document as a whole.
    fn whole(message: impl fmt::Display) -> Self {
        Self {
            path: String::new(),
            message: message.to_string(),
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
/// The values are JSON's: a tagged value, a number JSON does not write and
/// a key written twice are each refused by name.
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
/// and is refused in the parser's words.
pub(super) fn read_yaml(document: &[u8]) -> Result<Value, String> {
    let left = Left::whole();
    Weighed(&left)
        .deserialize(serde_yaml::Deserializer::from_slice(document))
        .map_err(|error| error.to_string())
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

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Value, E> {
        self.weigh(text.len())?;
        Ok(Value::String(text.to_owned()))
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

/// A map's key, read as its text: a key that looks like a number is its
/// digits.
struct Key<'a>(&'a Left);

impl<'de> DeserializeSeed<'de> for Key<'_> {
    type Value = String;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<String, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for Key<'_> {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a key of text")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<String, E> {
        self.0.hold(KEY)?;
        self.0.read(text.len())?;
        Ok(text.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    #[test]
    fn a_files_form_is_read_off_its_name() {
        for (name, format) in [
            ("policy.yaml", Format::Yaml),
            ("policy.yml", Format::Yaml),
            ("policy", Format::Yaml),
            ("policy.json", Format::Json),
            ("POLICY.JSON", Format::Json),
            ("..data/policy.json", Format::Json),
        ] {
            assert_eq!(Format::of(Path::new(name)), format, "{name}");
        }
        assert_eq!(
            (Format::Yaml.as_str(), Format::Json.as_str()),
            ("yaml", "json")
        );
    }

    #[test]
    fn a_form_is_named_as_it_is_written() {
        for format in [Format::Yaml, Format::Json] {
            assert_eq!(Format::named(format.as_str()), Some(format));
        }
        assert_eq!(Format::named("YAML"), None);
    }

    /// What `document` reads as, with the bytes it built and the text among
    /// them.
    fn weighed(document: &str) -> (Value, usize, usize) {
        let left = Left {
            built: Cell::new(usize::MAX),
            text: Cell::new(usize::MAX),
        };
        let read = Weighed(&left)
            .deserialize(serde_yaml::Deserializer::from_str(document))
            .expect("it reads");
        (
            read,
            usize::MAX - left.built.get(),
            usize::MAX - left.text.get(),
        )
    }

    /// The refusals of the read's two bounds.
    const PAST_TEXT: &str =
        "reads as more than 2 bytes of text for each byte of the size bound (131072)";

    /// A YAML document reads as the values JSON writes, a key as its text.
    /// Each value weighs what it holds: its own size and its text, a key
    /// its own size, its text and its place in the map, and a list keeps no
    /// room it does not fill. What JSON does not write is refused by name: a
    /// tagged value, a number past its range, a key written twice.
    #[test]
    fn a_yaml_document_reads_as_the_values_json_writes() {
        let read = read_yaml(
            b"a: [1, -1, 1.5, true, ~, text, \"7\"]\n1: {true: c}\ne: &anchor [x, y]\nf: *anchor\n",
        );
        let expected = serde_json::json!({
            "a": [1, -1, 1.5, true, null, "text", "7"],
            "1": {"true": "c"},
            "e": ["x", "y"],
            "f": ["x", "y"],
        });
        assert_eq!(read, Ok(expected));
        assert_eq!(read_yaml(b""), Ok(Value::Null));

        let (read, built, text) = weighed("{a,b}");
        assert_eq!(read, serde_json::json!({"a": null, "b": null}));
        assert_eq!(
            (built, text),
            (3 * NODE + 2 * (KEY + 1), 2),
            "a map and two nulls, under two keys of a byte"
        );
        let (read, built, text) = weighed("[a,[b]]");
        assert_eq!(read, serde_json::json!(["a", ["b"]]));
        assert_eq!(
            (built, text),
            (4 * NODE + 2, 2),
            "two lists and two texts of a byte"
        );
        let Value::Array(items) = read else {
            panic!("a list reads as a list")
        };
        assert_eq!(items.capacity(), 2, "a list keeps no room it does not fill");

        for (document, named) in [
            ("a: !tagged 1\n", "enum"),
            ("a: 340282366920938463463374607431768211455\n", "u128"),
            ("a: .inf\n", "is none JSON writes"),
            ("a: 1\na: 2\n", "the key `a` is written twice"),
        ] {
            match read_yaml(document.as_bytes()) {
                Err(message) => assert!(message.contains(named), "{document}: {message}"),
                Ok(read) => panic!("{document} read as {read}"),
            }
        }
    }

    /// A document reads as no more text than twice the size bound, its keys
    /// among it: the text an alias multiplies is refused by name where the
    /// bytes it builds are under their own bound, and text written out reads
    /// as it is, an escape of two bytes for a character of three too.
    #[test]
    fn a_yaml_document_reads_as_no_more_text_than_the_size_bound_may() {
        let refused = |document: String| match read_yaml(document.as_bytes()) {
            Err(message) => assert!(message.contains(PAST_TEXT), "{message}"),
            Ok(_) => panic!("{} bytes read", document.len()),
        };
        // A text of 5,000 bytes under 60 aliases: 305,000 bytes of text
        // where 262,144 may be read, and as much built where 10 MiB may be.
        let text = "x".repeat(5_000);
        let aliases = |name: &str, count: usize| vec![name; count].join(",");
        refused(format!("[&t {text},{}]", aliases("*t", 60)));
        // A map of a thousand keys of a hundred bytes under two aliases:
        // 300,000 bytes of keys, and 684,000 built.
        let mut keys = String::new();
        for key in 0..1000 {
            let _ = write!(keys, "{key:0>100}: 1,");
        }
        refused(format!("[&m {{{keys}}},{}]", aliases("*m", 2)));

        let escapes = "\\L".repeat(1000);
        let read = read_yaml(format!("\"{escapes}\"").as_bytes()).expect("it reads");
        assert_eq!(read.as_str().map(str::len), Some(3000));
    }
}
