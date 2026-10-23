//! A model family's marker tokens are spelled in up to four crates (the
//! tokenizer's renderer, Symphony's format and grammar tables, the two
//! legacy parsers), each copy right today. This test is the one place that
//! spells them: a revision of one copy fails here, naming the file, instead
//! of parting the renderer from the reader. The copies stay where they are
//! until the families' markers have a single source.

use std::{fs, path::Path};

/// One family's markers and the source files whose production code spells
/// each of them, in a code line (not a comment), as the string itself or
/// regex-escaped. A file's `#[cfg(test)]` tail and its comment lines are
/// not read, so neither a fixture nor a doc comment can stand in for the
/// copy the code runs, and a file that spells a marker only there is not
/// listed. A family with one such file pins nothing between crates and is
/// not listed either.
struct Family {
    name: &'static str,
    markers: &'static [&'static str],
    files: &'static [&'static str],
}

const FAMILIES: &[Family] = &[
    Family {
        name: "structured markup (open / close / sep)",
        markers: &["<|open|>", "<|close|>", "<|sep|>"],
        files: &[
            "crates/tokenizer/src/encoders/kimi_k3_xtml.rs",
            "crates/symphony/src/formats/kimi_k3.rs",
            "crates/symphony/src/grammar/xtml.rs",
            "crates/reasoning_parser/src/parsers/kimi_k3.rs",
            "crates/tool_parser/src/parsers/kimi_k3.rs",
        ],
    },
    Family {
        name: "inkling message markers",
        markers: &[
            "<|content_invoke_tool_json|>",
            "<|content_model_end_sampling|>",
            "<|message_model|>",
            "<|content_text|>",
            "<|content_thinking|>",
            "<|end_message|>",
        ],
        files: &[
            "crates/symphony/src/formats/inkling.rs",
            "crates/reasoning_parser/src/parsers/inkling.rs",
            "crates/tool_parser/src/parsers/inkling.rs",
        ],
    },
    Family {
        name: "inkling text tool call marker",
        markers: &["<|content_invoke_tool_text|>"],
        files: &[
            "crates/reasoning_parser/src/parsers/inkling.rs",
            "crates/tool_parser/src/parsers/inkling.rs",
        ],
    },
    Family {
        name: "gemma tool call tag",
        markers: &["<|tool_call>"],
        files: &[
            "crates/symphony/src/formats/gemma4.rs",
            "crates/reasoning_parser/src/parsers/gemma4.rs",
            "crates/tool_parser/src/parsers/gemma4.rs",
        ],
    },
    Family {
        name: "minimax tool namespace",
        markers: &["]<]minimax[>["],
        files: &[
            "crates/symphony/src/formats/minimax_m3.rs",
            "crates/tool_parser/src/parsers/minimax_m3.rs",
        ],
    },
];

/// The code lines of a source file before its `#[cfg(test)]` module: what
/// the crate ships, without the fixtures its tests spell and without the
/// comments that describe the markers.
fn production_code(source: &str) -> String {
    let production = match source.find("\n#[cfg(test)]") {
        Some(at) => &source[..at],
        None => source,
    };
    production
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every copy of every marker of every family is spelled as the family's
/// table says, in a code line of the file that defines it: the string
/// itself or its regex-escaped form.
#[test]
fn every_copy_of_a_family_marker_spells_the_same_string() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut missing = Vec::new();
    for family in FAMILIES {
        for file in family.files {
            let source =
                fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
            let production = production_code(&source);
            for marker in family.markers {
                let escaped = regex::escape(marker);
                if !production.contains(marker) && !production.contains(&escaped) {
                    missing.push(format!("{}: {file} does not spell {marker}", family.name));
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "a family's marker differs between its copies:\n{}",
        missing.join("\n")
    );
}

/// No family, file or marker is listed twice, so the table's counts mean
/// what they say.
#[test]
fn the_marker_table_has_no_repeated_entry() {
    for (i, family) in FAMILIES.iter().enumerate() {
        assert!(
            !FAMILIES[..i].iter().any(|f| f.name == family.name),
            "{} listed twice",
            family.name
        );
        for (i, file) in family.files.iter().enumerate() {
            assert!(
                !family.files[..i].contains(file),
                "{}: {file} listed twice",
                family.name
            );
        }
        for (i, marker) in family.markers.iter().enumerate() {
            assert!(
                !family.markers[..i].contains(marker),
                "{}: {marker} listed twice",
                family.name
            );
        }
    }
}
