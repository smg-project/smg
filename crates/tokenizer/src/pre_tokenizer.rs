//! What the load reads of a pipeline's pre-tokenizer before handing the
//! file to the backend.

use serde_json::Value;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;

/// The first `Split` pattern of `pre_tokenizer` (through a `Sequence`)
/// anchored to the piece it runs on: `\A`, `\z`, `\Z`, `\G`, or `^` and `$`
/// outside a class. `tokenizers` runs each `Split` on each piece the steps
/// before it cut, where such an anchor holds at the piece's ends; the
/// backend splits a run of seven or more digits otherwise under one (a
/// checkpoint grouping digits by threes with
/// `\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)`), so the load keeps the pipeline for
/// such a file. The pre-tokenizer is read through its serialized form, the
/// crate keeping the pattern private.
pub(crate) fn anchored_split_pattern(
    pre_tokenizer: Option<&PreTokenizerWrapper>,
) -> Option<String> {
    fn walk(step: &Value) -> Option<String> {
        match step.get("type").and_then(Value::as_str) {
            Some("Sequence") => step
                .get("pretokenizers")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find_map(walk),
            Some("Split") => step
                .get("pattern")
                .and_then(|pattern| pattern.get("Regex"))
                .and_then(Value::as_str)
                .filter(|pattern| pattern_has_piece_anchor(pattern))
                .map(str::to_owned),
            _ => None,
        }
    }
    let serialized = serde_json::to_value(pre_tokenizer?).ok()?;
    walk(&serialized)
}

/// Whether `pattern` anchors to the start or the end of its subject: the
/// escapes `\A`, `\z`, `\Z` and `\G`, or `^` and `$` outside a character
/// class (`[^...]` negates a class, so it is not an anchor).
pub(crate) fn pattern_has_piece_anchor(pattern: &str) -> bool {
    let mut chars = pattern.chars();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('A' | 'z' | 'Z' | 'G') if !in_class => return true,
                _ => {}
            },
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '^' | '$' if !in_class => return true,
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The GPT-2 and Qwen patterns (a lookahead, a negated class) carry no
    /// anchor; a pattern anchored to the piece does, escaped or bare.
    #[test]
    fn a_pattern_anchored_to_the_piece_is_told_from_one_that_is_not() {
        for pattern in [
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            r"\\A[\^$]",
        ] {
            assert!(!pattern_has_piece_anchor(pattern), "{pattern}");
        }
        for pattern in [
            r"\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)",
            r"\A\p{Nd}{3}|\G\p{Nd}{3}",
            r"\p{Nd}{1,510}(?=(?>\p{Nd}{510})*(?:\P{Nd}|$))",
            r"^\s+",
        ] {
            assert!(pattern_has_piece_anchor(pattern), "{pattern}");
        }
    }

    /// The anchored pattern is found through a `Sequence`, and a
    /// pre-tokenizer without one, or none, gives nothing.
    #[test]
    fn the_anchored_pattern_of_a_pipeline_is_found_through_a_sequence() {
        let anchored: PreTokenizerWrapper = serde_json::from_value(json!({
            "type": "Sequence", "pretokenizers": [
                {"type": "Digits", "individual_digits": false},
                {"type": "Split", "pattern": {"Regex": "\\A\\p{Nd}{3}|\\G\\p{Nd}{3}"}, "behavior": "Isolated", "invert": false}]
        }))
        .unwrap();
        assert_eq!(
            anchored_split_pattern(Some(&anchored)).as_deref(),
            Some(r"\A\p{Nd}{3}|\G\p{Nd}{3}")
        );
        let byte_level: PreTokenizerWrapper = serde_json::from_value(json!({
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true
        }))
        .unwrap();
        assert_eq!(anchored_split_pattern(Some(&byte_level)), None);
        assert_eq!(anchored_split_pattern(None), None);
    }
}
