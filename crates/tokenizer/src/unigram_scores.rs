//! The Unigram section of a tokenizer file, read with the scores the
//! reference reads.
//!
//! A Unigram model splits a stretch of text where its pieces' scores add up
//! highest. A long run of one character has many splits whose sums are equal
//! or one bit apart, so the split that wins rests on the last bit of every
//! score: on how the decimal text of a score becomes a float. The
//! `tokenizers` crate leaves that to `serde_json`, and the reference's own
//! builds take `serde_json` as it comes: the digits of a number as one
//! integer, scaled by one power of ten, which lands one bit beside the
//! nearest float for many 17-digit scores. Built with `float_roundtrip`,
//! which this crate asks for so that a tool call's numbers render exactly,
//! the same parser reads the nearest float, ties fall another way, and ids
//! differ from the reference's. So this crate reads the section itself:
//! every score by the plain parser's rule, whatever features the build's
//! parser has.

use std::fmt;

use serde::{
    de::{
        self, DeserializeSeed, Error as _, IgnoredAny, MapAccess, SeqAccess, Unexpected, Visitor,
    },
    Deserialize, Deserializer,
};
use serde_json::{value::RawValue, Value};
use tokenizers::models::{unigram::Unigram, ModelWrapper};

/// Reads a Unigram section from the entries of `map`: what the caller read
/// of the section before it knew the kind, put back in front, then the
/// rest. `vocab` is the vocabulary when the caller has read it already
/// ([`VocabShape`]). These are the entries the `tokenizers` crate's reader
/// takes (`unk_id`, `vocab`, `byte_fallback`; any other is skipped), handed
/// to the crate's own builder, so the model is the one that crate builds
/// from the same file when its parser has no `float_roundtrip`.
pub(crate) fn section<'de, A: MapAccess<'de>>(
    mut vocab: Option<Vec<(String, Score)>>,
    mut map: A,
) -> Result<ModelWrapper, A::Error> {
    let mut unk_id: Option<usize> = None;
    let mut byte_fallback = false;
    while let Some(key) = map.next_key::<String>()? {
        match key.as_str() {
            "unk_id" => unk_id = map.next_value()?,
            "byte_fallback" => byte_fallback = map.next_value()?,
            "vocab" => vocab = Some(map.next_value()?),
            "type" => {
                let kind: String = map.next_value()?;
                if kind != "Unigram" {
                    return Err(A::Error::invalid_value(Unexpected::Str(&kind), &"Unigram"));
                }
            }
            _ => {
                map.next_value::<IgnoredAny>()?;
            }
        }
    }
    let vocab = vocab
        .ok_or_else(|| A::Error::custom("Missing vocab"))?
        .into_iter()
        .map(|(piece, Score(score))| (piece, score))
        .collect();
    Unigram::from(vocab, unk_id, byte_fallback)
        .map(ModelWrapper::Unigram)
        .map_err(A::Error::custom)
}

/// The Unigram model of a tokenizer file, its section read by [`section`]
/// whatever the order of its keys. It is for a file whose section the
/// `tokenizers` crate's reader took, which reads the scores by its own rule:
/// the loader asks for the section once more, from the text of the file.
pub(crate) fn of_file(content: &str) -> serde_json::Result<ModelWrapper> {
    struct Model(ModelWrapper);

    impl<'de> Deserialize<'de> for Model {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct Reader;

            impl<'de> Visitor<'de> for Reader {
                type Value = ModelWrapper;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a Unigram model section")
                }

                fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<ModelWrapper, A::Error> {
                    section(None, map)
                }
            }

            deserializer.deserialize_map(Reader).map(Model)
        }
    }

    #[derive(Deserialize)]
    struct File {
        model: Model,
    }

    serde_json::from_str::<File>(content).map(|file| file.model.0)
}

/// The value of a `vocab` entry read before the section's kind is known.
pub(crate) enum Vocab {
    /// A list of pieces with their scores: a Unigram section's, each score
    /// read by [`read`].
    Scored(Vec<(String, Score)>),
    /// Any other vocabulary, kept for the crate's reader.
    Other(Value),
}

/// Reads a `vocab` entry by its shape (see [`Vocab`]).
pub(crate) struct VocabShape;

impl<'de> DeserializeSeed<'de> for VocabShape {
    type Value = Vocab;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Vocab, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for VocabShape {
    type Value = Vocab;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a vocabulary")
    }

    fn visit_seq<S: SeqAccess<'de>>(self, seq: S) -> Result<Vocab, S::Error> {
        Vec::deserialize(de::value::SeqAccessDeserializer::new(seq)).map(Vocab::Scored)
    }

    fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Vocab, M::Error> {
        Value::deserialize(de::value::MapAccessDeserializer::new(map)).map(Vocab::Other)
    }
}

/// One score of the vocabulary, read from its text by [`read`].
pub(crate) struct Score(f64);

impl<'de> Deserialize<'de> for Score {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = Box::<RawValue>::deserialize(deserializer)?;
        read(text.get()).map(Score).ok_or_else(|| {
            D::Error::custom(format_args!(
                "a Unigram score that is no number in range: {}",
                text.get()
            ))
        })
    }
}

/// A JSON number as `serde_json` reads it without `float_roundtrip` (its
/// `parse_integer`, `parse_decimal`, `parse_exponent` and `f64_from_parts`,
/// step for step): the digits gathered into one `u64` for as long as they
/// fit, the rest dropped, and that integer scaled by one power of ten, a
/// multiplication or a division that rounds a second time. `None` is a text
/// that parser refuses: no number, or one out of an `f64`'s range.
fn read(text: &str) -> Option<f64> {
    let (negative, text) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let (text, written) = match text.split_once(['e', 'E']) {
        Some((text, written)) => (text, Some(written)),
        None => (text, None),
    };
    let (whole, fraction) = match text.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (text, None),
    };
    let padded = whole.len() > 1 && whole.starts_with('0');
    if !digits_only(whole) || padded || !fraction.is_none_or(digits_only) {
        return None;
    }

    let mut significand = 0u64;
    let mut exponent = 0i32;
    for (at, digit) in whole.bytes().enumerate() {
        match gathered(significand, digit) {
            Some(more) => significand = more,
            // A whole part too long for the integer: this digit and every
            // one after it only scale the number.
            None => {
                exponent = i32::try_from(whole.len() - at).ok()?;
                break;
            }
        }
    }
    for digit in fraction.unwrap_or_default().bytes() {
        match gathered(significand, digit) {
            Some(more) => {
                significand = more;
                exponent -= 1;
            }
            // The first digit that does not fit ends the fraction.
            None => break,
        }
    }
    if let Some(written) = written {
        let (grows, written) = match written.strip_prefix('-') {
            Some(rest) => (false, rest),
            None => (true, written.strip_prefix('+').unwrap_or(written)),
        };
        if !digits_only(written) {
            return None;
        }
        let mut by = 0i32;
        for digit in written.bytes() {
            let more = by
                .checked_mul(10)
                .and_then(|by| by.checked_add(i32::from(digit - b'0')));
            match more {
                Some(more) => by = more,
                // An exponent too long for its integer: zero for a number
                // that is zero or shrinks, out of range for one that grows.
                None if significand == 0 || !grows => return Some(signed(negative, 0.0)),
                None => return None,
            }
        }
        exponent = if grows {
            exponent.saturating_add(by)
        } else {
            exponent.saturating_sub(by)
        };
    }

    let mut value = significand as f64;
    loop {
        match POWERS_OF_TEN.get(exponent.unsigned_abs() as usize) {
            Some(&power) if exponent >= 0 => {
                value *= power;
                if value.is_infinite() {
                    return None;
                }
                break;
            }
            Some(&power) => {
                value /= power;
                break;
            }
            // Beyond the table: zero stays zero, a growing number is out
            // of range, a shrinking one is divided down in steps.
            None if value == 0.0 => break,
            None if exponent >= 0 => return None,
            None => {
                value /= 1e308;
                exponent += 308;
            }
        }
    }
    Some(signed(negative, value))
}

/// Whether the text is one or more ASCII digits.
fn digits_only(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

/// The integer with one more digit, `None` when it no longer fits.
fn gathered(significand: u64, digit: u8) -> Option<u64> {
    significand
        .checked_mul(10)?
        .checked_add(u64::from(digit - b'0'))
}

fn signed(negative: bool, value: f64) -> f64 {
    if negative {
        -value
    } else {
        value
    }
}

/// The powers of ten an `f64` holds, each the float nearest to it: the
/// scale of [`read`].
#[rustfmt::skip]
const POWERS_OF_TEN: [f64; 309] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
    1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
    1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29,
    1e30, 1e31, 1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38, 1e39,
    1e40, 1e41, 1e42, 1e43, 1e44, 1e45, 1e46, 1e47, 1e48, 1e49,
    1e50, 1e51, 1e52, 1e53, 1e54, 1e55, 1e56, 1e57, 1e58, 1e59,
    1e60, 1e61, 1e62, 1e63, 1e64, 1e65, 1e66, 1e67, 1e68, 1e69,
    1e70, 1e71, 1e72, 1e73, 1e74, 1e75, 1e76, 1e77, 1e78, 1e79,
    1e80, 1e81, 1e82, 1e83, 1e84, 1e85, 1e86, 1e87, 1e88, 1e89,
    1e90, 1e91, 1e92, 1e93, 1e94, 1e95, 1e96, 1e97, 1e98, 1e99,
    1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107, 1e108, 1e109,
    1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
    1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129,
    1e130, 1e131, 1e132, 1e133, 1e134, 1e135, 1e136, 1e137, 1e138, 1e139,
    1e140, 1e141, 1e142, 1e143, 1e144, 1e145, 1e146, 1e147, 1e148, 1e149,
    1e150, 1e151, 1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
    1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167, 1e168, 1e169,
    1e170, 1e171, 1e172, 1e173, 1e174, 1e175, 1e176, 1e177, 1e178, 1e179,
    1e180, 1e181, 1e182, 1e183, 1e184, 1e185, 1e186, 1e187, 1e188, 1e189,
    1e190, 1e191, 1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199,
    1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207, 1e208, 1e209,
    1e210, 1e211, 1e212, 1e213, 1e214, 1e215, 1e216, 1e217, 1e218, 1e219,
    1e220, 1e221, 1e222, 1e223, 1e224, 1e225, 1e226, 1e227, 1e228, 1e229,
    1e230, 1e231, 1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239,
    1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247, 1e248, 1e249,
    1e250, 1e251, 1e252, 1e253, 1e254, 1e255, 1e256, 1e257, 1e258, 1e259,
    1e260, 1e261, 1e262, 1e263, 1e264, 1e265, 1e266, 1e267, 1e268, 1e269,
    1e270, 1e271, 1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279,
    1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287, 1e288, 1e289,
    1e290, 1e291, 1e292, 1e293, 1e294, 1e295, 1e296, 1e297, 1e298, 1e299,
    1e300, 1e301, 1e302, 1e303, 1e304, 1e305, 1e306, 1e307, 1e308,
];

#[cfg(test)]
mod tests {
    use tokenizers::models::ModelWrapper;

    use super::read;
    use crate::{huggingface::HuggingFaceTokenizer, traits::Encoder, word_cache::LateBoundModel};

    /// Each text with the float `serde_json` reads from it when built
    /// without `float_roundtrip`, as bits; `None` is a number it refuses.
    /// The values are that parser's own, read off a build of the
    /// `tokenizers` crate that has it so. The twelfth is one bit beside the
    /// nearest float; the fifteenth takes a fraction digit after it dropped
    /// a whole one.
    const READ: &[(&str, Option<u64>)] = &[
        ("0", Some(0x0000_0000_0000_0000)),
        ("-0", Some(0x8000_0000_0000_0000)),
        ("7", Some(0x401c_0000_0000_0000)),
        ("-12", Some(0xc028_0000_0000_0000)),
        ("0.5", Some(0x3fe0_0000_0000_0000)),
        ("-2.5", Some(0xc004_0000_0000_0000)),
        ("123.456", Some(0x405e_dd2f_1a9f_be77)),
        ("0.000001", Some(0x3eb0_c6f7_a0b5_ed8d)),
        ("1e3", Some(0x408f_4000_0000_0000)),
        ("1E-2", Some(0x3f84_7ae1_47ae_147b)),
        ("1.5e+2", Some(0x4062_c000_0000_0000)),
        ("-1.1695698339729451", Some(0xbff2_b68e_dbb4_d4c9)),
        ("0.12345678901234567890123", Some(0x3fbf_9add_3746_f65f)),
        (
            "123456789012345678901234567890",
            Some(0x45f8_ee90_ff6c_373e),
        ),
        ("18446744073709551616.3", Some(0x43f0_0000_0000_0000)),
        ("1e22", Some(0x4480_f0cf_064d_d592)),
        ("1e23", Some(0x44b5_2d02_c7e1_4af6)),
        ("1.7976931348623157e308", Some(0x7fef_ffff_ffff_ffff)),
        ("1e-320", Some(0x0000_0000_0000_07e8)),
        ("5e-324", Some(0x0000_0000_0000_0001)),
        ("1e400", None),
        ("0e999999999999", Some(0x0000_0000_0000_0000)),
        ("1e-999999999999", Some(0x0000_0000_0000_0000)),
        ("1e999999999999", None),
        ("-1.5e-400", Some(0x8000_0000_0000_0000)),
    ];

    /// A Unigram section in which "a" twice scores exactly what "aa" scores
    /// under the plain rule: the second literal is twice the float the first
    /// reads as. The nearest float to the first literal is one bit closer
    /// to zero, and read that way "a" twice scores higher.
    const TIED: &str = r#"{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["a",-1.1695698339729451],["aa",-2.3391396679458905]],"byte_fallback":false}"#;

    /// The same section in the form of a file written before the
    /// `tokenizers` crate named the kind: no `type` entry.
    const TIED_OLD: &str = r#"{"unk_id":0,"vocab":[["<unk>",0.0],["a",-1.1695698339729451],["aa",-2.3391396679458905]]}"#;

    /// Loads a tokenizer file that holds nothing but the model section.
    fn load(model: &str) -> Result<HuggingFaceTokenizer, String> {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let path = dir.path().join("tokenizer.json");
        let file = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{model}}}"#
        );
        std::fs::write(&path, file).expect("the file is written");
        HuggingFaceTokenizer::from_file(path.to_str().expect("a UTF-8 path"))
            .map_err(|error| error.to_string())
    }

    fn ids(tokenizer: &HuggingFaceTokenizer, text: &str) -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .expect("the text encodes")
            .token_ids()
            .to_vec()
    }

    #[test]
    fn a_number_reads_as_the_plain_parser_reads_it() {
        for &(text, bits) in READ {
            assert_eq!(read(text).map(f64::to_bits), bits, "{text}");
        }
    }

    #[test]
    fn what_is_no_number_is_refused() {
        for text in [
            "", "-", "null", "true", "\"1\"", "[1]", "1.", ".5", "1e", "1e+", "0x10", "1 ", "+1",
            "01", "-00.5",
        ] {
            assert_eq!(read(text), None, "{text:?}");
        }
    }

    #[test]
    fn the_scores_of_a_section_are_the_plain_rules() {
        for section in [TIED, TIED_OLD] {
            scores_of(section);
        }
    }

    fn scores_of(section: &str) {
        let model: LateBoundModel = serde_json::from_str(section).expect("the section reads");
        let ModelWrapper::Unigram(unigram) = ModelWrapper::from(model) else {
            panic!("the section builds a Unigram model");
        };
        let scores: Vec<u64> = unigram.iter().map(|(_, score)| score.to_bits()).collect();
        assert_eq!(
            scores,
            [0, 0xbff2_b68e_dbb4_d4c9, 0xc002_b68e_dbb4_d4c9],
            "the reference reads the second score one bit beside the nearest float"
        );
    }

    #[test]
    fn a_tie_under_the_plain_rule_stays_a_tie() {
        for section in [TIED, TIED_OLD] {
            let tokenizer = load(section).expect("the file loads");
            // The model keeps the longer piece on a tie, as the reference does.
            assert_eq!(ids(&tokenizer, "aa"), [2], "{section}");
            assert_eq!(ids(&tokenizer, "aaa"), [1, 2], "{section}");
        }
    }

    #[test]
    fn a_key_no_section_has_does_not_change_the_scores_wherever_it_stands() {
        let in_front = TIED.replacen('{', r#"{"metadata":{},"#, 1);
        let behind = TIED.replacen(
            r#""type":"Unigram","#,
            r#""type":"Unigram","metadata":{},"#,
            1,
        );
        let in_front_of_an_old_file = TIED_OLD.replacen('{', r#"{"metadata":{},"#, 1);
        for section in [in_front, behind, in_front_of_an_old_file] {
            assert!(section.contains("metadata"), "{section}");
            let tokenizer = load(&section).expect("the file loads");
            assert_eq!(ids(&tokenizer, "aa"), [2], "{section}");
            assert_eq!(ids(&tokenizer, "aaa"), [1, 2], "{section}");
        }
    }

    #[test]
    fn a_key_of_another_kind_leaves_the_scores_of_a_section_that_names_its_kind() {
        for entry in [
            r#""dropout":null"#,
            r#""unk_token":"<unk>""#,
            r#""continuing_subword_prefix":"""#,
            r#""end_of_word_suffix":"""#,
            r#""fuse_unk":false"#,
            r#""ignore_merges":false"#,
            r#""merges":[]"#,
            r#""max_input_chars_per_word":100"#,
        ] {
            let in_front = TIED.replacen('{', &format!("{{{entry},"), 1);
            let behind = TIED.replacen(
                r#""type":"Unigram","#,
                &format!(r#""type":"Unigram",{entry},"#),
                1,
            );
            for section in [in_front, behind] {
                assert!(section.contains(entry), "{section}");
                let tokenizer = load(&section).expect("the file loads");
                assert_eq!(ids(&tokenizer, "aa"), [2], "{section}");
                assert_eq!(ids(&tokenizer, "aaa"), [1, 2], "{section}");
            }
        }
    }

    #[test]
    fn merges_in_front_of_type_leave_the_scores_of_a_unigram_section() {
        // The merges hand the section to the `tokenizers` reader: behind a
        // key of another kind, and alone.
        for section in [
            TIED.replacen('{', r#"{"dropout":null,"merges":[],"#, 1),
            TIED.replacen('{', r#"{"merges":[["a","a"]],"#, 1),
        ] {
            let tokenizer = load(&section).expect("the file loads");
            assert_eq!(ids(&tokenizer, "aa"), [2], "{section}");
            assert_eq!(ids(&tokenizer, "aaa"), [1, 2], "{section}");
        }
    }

    #[test]
    fn a_vocabulary_in_front_of_type_leaves_the_scores_of_a_unigram_section() {
        // Behind a key of another kind the vocabulary hands the section to
        // the `tokenizers` reader: with `type` behind the vocabulary, and
        // with no `type` at all.
        let vocab = ENTRIES[2];
        for section in [
            format!(r#"{{"unk_token":"<unk>","unk_id":0,{vocab},"type":"Unigram"}}"#),
            format!(r#"{{"dropout":null,"unk_id":0,{vocab}}}"#),
        ] {
            let tokenizer = load(&section).expect("the file loads");
            assert_eq!(ids(&tokenizer, "aa"), [2], "{section}");
            assert_eq!(ids(&tokenizer, "aaa"), [1, 2], "{section}");
        }
    }

    /// The entries of [`TIED`], one by one.
    const ENTRIES: [&str; 4] = [
        r#""type":"Unigram""#,
        r#""unk_id":0"#,
        r#""vocab":[["<unk>",0.0],["a",-1.1695698339729451],["aa",-2.3391396679458905]]"#,
        r#""byte_fallback":false"#,
    ];

    /// The section made of `entries`, once in each of their orders.
    fn orders(entries: &[&str]) -> Vec<String> {
        fn extend(order: &mut Vec<usize>, entries: usize, all: &mut Vec<Vec<usize>>) {
            if order.len() == entries {
                all.push(order.clone());
                return;
            }
            for entry in 0..entries {
                if !order.contains(&entry) {
                    order.push(entry);
                    extend(order, entries, all);
                    order.pop();
                }
            }
        }
        let mut all = Vec::new();
        extend(&mut Vec::new(), entries.len(), &mut all);
        all.iter()
            .map(|order| {
                let section: Vec<&str> = order.iter().map(|&entry| entries[entry]).collect();
                format!("{{{}}}", section.join(","))
            })
            .collect()
    }

    #[test]
    fn a_section_in_any_key_order_has_the_same_scores() {
        // The four entries of a section, then the three and the two of a
        // file written before the kind was named.
        let named = orders(&ENTRIES);
        let unnamed = orders(&ENTRIES[1..]);
        let oldest = orders(&ENTRIES[1..3]);
        assert_eq!((named.len(), unnamed.len(), oldest.len()), (24, 6, 2));
        for section in named.iter().chain(&unnamed).chain(&oldest) {
            scores_of(section);
            let tokenizer = load(section).expect("the file loads");
            assert_eq!(ids(&tokenizer, "aa"), [2], "{section}");
        }
    }

    #[test]
    fn a_section_that_cannot_be_read_says_why() {
        for (model, why) in [
            (r#"{"type":"Unigram","unk_id":0}"#, "Missing vocab"),
            (
                r#"{"unk_id":0,"vocab":[["<unk>",0.0]],"type":"BPE"}"#,
                "expected Unigram",
            ),
            (
                r#"{"type":"Unigram","unk_id":0,"vocab":[["<unk>","0.0"]]}"#,
                "a Unigram score that is no number in range: \"0.0\"",
            ),
            (
                r#"{"type":"Unigram","unk_id":0,"vocab":[["<unk>",1e400]]}"#,
                "a Unigram score that is no number in range: 1e400",
            ),
        ] {
            let error = load(model).err().expect("the file is refused");
            assert!(error.contains(why), "{model}: {error}");
        }
    }
}
