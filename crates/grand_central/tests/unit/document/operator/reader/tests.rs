use std::fmt::Write as _;

use super::*;
use crate::document::spec::ScoreSpec;

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
/// value under a local tag, a number that is not finite, a key written
/// twice. A global tag is read as YAML reads it: `!!str` leaves a number
/// its text, `!!int` types a quoted one, and `!!null` refuses a text
/// that is none. A whole number past 64 bits reads as a fraction.
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

    let long = read_yaml(b"a: 340282366920938463463374607431768211455\nb: -18446744073709551616\n");
    let fractions = serde_json::json!({"a": u128::MAX as f64, "b": -(2f64.powi(64))});
    assert_eq!(long, Ok(fractions));

    let global = read_yaml(b"a: !!str 1\nb: !!int \"5\"\n");
    assert_eq!(global, Ok(serde_json::json!({"a": "1", "b": 5})));

    for (document, named) in [
        (
            "a: !tagged 1\n",
            "the tag `!tagged` is none JSON writes: take the tag out",
        ),
        (
            "a: !!null x\n",
            "invalid value: string \"x\", expected null",
        ),
        ("a: .inf\n", "is none JSON writes"),
        ("a: 1\na: 2\n", "the key `a` is written twice"),
    ] {
        match read_yaml(document.as_bytes()) {
            Err(message) => assert!(message.contains(named), "{document}: {message}"),
            Ok(read) => panic!("{document} read as {read}"),
        }
    }
}

/// A local tag is refused by its name wherever it stands, on a key as on
/// a value: at the root, in a nested map, in a flow map, on a key that
/// is a list or a map, with an anchor on either side of the tag, on a
/// key with nothing under it, and as `!` alone. A handle that a `%TAG`
/// line resolves to a local name is refused by that name. The name is the
/// tag's letters as they are written.
#[test]
fn a_local_tag_is_refused_by_its_name_on_a_key_as_on_a_value() {
    for (document, tag) in [
        ("!mine a: 1\n", "!mine"),
        ("a:\n  !mine b: 1\n", "!mine"),
        ("{!mine a: 1}\n", "!mine"),
        ("a: [{!mine b: 1}]\n", "!mine"),
        ("? !mine [a]\n: 1\n", "!mine"),
        ("? !mine {a: 1}\n: 2\n", "!mine"),
        ("!mine &k a: 1\n", "!mine"),
        ("&k !mine a: 1\n", "!mine"),
        ("!mine \"a\": 1\n", "!mine"),
        ("!mine a:\n", "!mine"),
        ("!mine ~: 1\n", "!mine"),
        ("!MiNe a: 1\n", "!MiNe"),
        ("! a: 1\n", "!"),
        ("%TAG !e! !my-\n---\n!e!x a: 1\n", "!my-x"),
        // The same tag where it was refused before: a value, a list, an
        // item, a map and the document.
        ("a: !mine 1\n", "!mine"),
        ("a: !mine [1]\n", "!mine"),
        ("a: [!mine 1]\n", "!mine"),
        ("a: !mine {b: 1}\n", "!mine"),
        ("!mine {a: 1}\n", "!mine"),
        ("a: !MiNe 1\n", "!MiNe"),
        ("a: ! 1\n", "!"),
        ("%TAG !e! !my-\n---\na: !e!x 1\n", "!my-x"),
    ] {
        let named = format!("the tag `{tag}` is none JSON writes: take the tag out");
        match read_yaml(document.as_bytes()) {
            Err(refused) => assert!(refused.contains(&named), "{document}: {refused}"),
            Ok(read) => panic!("{document} read as {read}"),
        }
    }
}

/// A global tag is read as YAML reads it, by the parser and ahead of
/// this read. One of the four that type a scalar types it, or refuses
/// it where its text is none. Any other reads a scalar as its text, a
/// tag written whole and one a `%TAG` line makes global among them. On
/// a list, a map or a key a global tag changes nothing.
#[test]
fn a_global_tag_is_read_as_yaml_reads_it() {
    for (document, read) in [
        ("a: !!int \"5\"\n", serde_json::json!({"a": 5})),
        ("a: !!float 5\n", serde_json::json!({"a": 5.0})),
        ("a: !!bool \"true\"\n", serde_json::json!({"a": true})),
        ("a: !!null ~\n", serde_json::json!({"a": null})),
        ("a: !!str 1\n", serde_json::json!({"a": "1"})),
        ("a: !!binary YQ==\n", serde_json::json!({"a": "YQ=="})),
        (
            "a: !<tag:example.com,2000:x> 1\n",
            serde_json::json!({"a": "1"}),
        ),
        (
            "%TAG ! tag:example.com,2000:\n---\na: !mine 1\n",
            serde_json::json!({"a": "1"}),
        ),
        (
            "%TAG !e! tag:example.com,2000:\n---\na: !e!x 1\n",
            serde_json::json!({"a": "1"}),
        ),
        ("a: !!int [1]\n", serde_json::json!({"a": [1]})),
        ("a: !!bool {b: 1}\n", serde_json::json!({"a": {"b": 1}})),
        ("!!str a: 1\n", serde_json::json!({"a": 1})),
        ("!!int 1: x\n", serde_json::json!({"1": "x"})),
        (
            "!<tag:example.com,2000:x> a: 1\n",
            serde_json::json!({"a": 1}),
        ),
        (
            "%TAG ! tag:example.com,2000:\n---\n!mine a: 1\n",
            serde_json::json!({"a": 1}),
        ),
        (
            "%TAG !e! tag:example.com,2000:\n---\n!e!x a: 1\n",
            serde_json::json!({"a": 1}),
        ),
    ] {
        assert_eq!(read_yaml(document.as_bytes()), Ok(read), "{document}");
    }
    for (document, named) in [
        (
            "a: !!int five\n",
            "invalid value: string \"five\", expected an integer",
        ),
        (
            "a: !!null \"\"\n",
            "invalid value: string \"\", expected null",
        ),
    ] {
        match read_yaml(document.as_bytes()) {
            Err(refused) => assert!(refused.contains(named), "{document}: {refused}"),
            Ok(read) => panic!("{document} read as {read}"),
        }
    }
}

/// A key with no local tag reads as it did where the parser's string
/// entry read it. A key that looks like a number, a boolean or a null is
/// its text as written, plain and quoted, an empty key too. A key by an
/// alias is its anchor's text. A key weighs its own size, its text and
/// its place in the map, under a global tag too. The JSON form reads a
/// key the same way, and a key that begins with `!` there is its text.
/// A key that is a list or a map is refused at its own place, in the
/// words the string entry had for it: it wants a key of text, not a tag.
#[test]
fn a_key_with_no_local_tag_reads_as_its_text() {
    let texts = read_yaml(
        b"1: a\n007: b\n0x10: c\ntrue: d\n~: e\n\"1.5\": f\n'false': g\nnull: h\n-0: i\n\"\": j\n",
    );
    let expected = serde_json::json!({
        "1": "a", "007": "b", "0x10": "c", "true": "d", "~": "e",
        "1.5": "f", "false": "g", "null": "h", "-0": "i", "": "j",
    });
    assert_eq!(texts, Ok(expected));
    assert_eq!(read_yaml(b"? \n: a\n"), Ok(serde_json::json!({"": "a"})));
    assert_eq!(
        read_yaml(b"&k a: 1\nb: {*k : 2}\nc: &v x\nd: {*v : 3}\n"),
        Ok(serde_json::json!({"a": 1, "b": {"a": 2}, "c": "x", "d": {"x": 3}}))
    );

    for document in ["{a: ~}", "{!!str a: ~}", "{&k a: ~}"] {
        let (read, built, text) = weighed(document);
        assert_eq!(read, serde_json::json!({"a": null}), "{document}");
        assert_eq!((built, text), (2 * NODE + KEY + 1, 1), "{document}");
    }

    let json = read_json(br#"{"1": 1, "true": 2, "": 3, "!mine a": 4}"#);
    assert_eq!(
        json.map_err(|refused| refused.to_string()),
        Ok(serde_json::json!({"1": 1, "true": 2, "": 3, "!mine a": 4}))
    );

    for (document, refused) in [
        (
            "? [a, b]\n: 1\n",
            "invalid type: sequence, expected a key of text at line 1 column 3",
        ),
        (
            "? {a: 1}\n: 2\n",
            "invalid type: map, expected a key of text at line 1 column 3",
        ),
        (
            "{[a]: 1}\n",
            "invalid type: sequence, expected a key of text at line 1 column 2",
        ),
        (
            "a: &k [x]\nb: {*k : 2}\n",
            "b: invalid type: sequence, expected a key of text at line 1 column 4",
        ),
    ] {
        assert_eq!(
            read_yaml(document.as_bytes()),
            Err(refused.to_owned()),
            "{document}"
        );
    }
    // The rewording is by the parser's phrase: a document that writes
    // the phrase as a key has it reworded where a refusal quotes the key.
    let phrase = "invalid type: map, expected a YAML tag starting with '!'";
    assert_eq!(
        read_yaml(format!("{{\"{phrase}\": 1, \"{phrase}\": 2}}").as_bytes()),
        Err("the key `invalid type: map, expected a key of text` is written twice".to_owned())
    );
}

/// The parser bounds a nest in its own words. A list reads 128 deep and
/// no deeper. A map that holds a key reads 127 deep, as every nest of the
/// JSON form does: the read of a key holds one level of the parser's
/// count. A map that holds none reads 128 deep, as a list does.
#[test]
fn the_parser_reads_a_list_128_deep_and_a_map_127_deep() {
    let lists = |depth: usize| format!("{}{}", "[".repeat(depth), "]".repeat(depth));
    let maps = |depth: usize| format!("{}1{}", "{a: ".repeat(depth), "}".repeat(depth));
    assert!(read_yaml(lists(128).as_bytes()).is_ok());
    assert_eq!(
        read_yaml(lists(129).as_bytes()),
        Err("recursion limit exceeded at line 1 column 129".to_owned())
    );
    assert!(read_yaml(maps(127).as_bytes()).is_ok());
    assert_eq!(
        read_yaml(maps(128).as_bytes()),
        Err("recursion limit exceeded at line 1 column 510".to_owned())
    );
    let json = |depth: usize| format!("{}1{}", "{\"a\":".repeat(depth), "}".repeat(depth));
    assert!(read_json(json(127).as_bytes()).is_ok());
    assert!(read_json(json(128).as_bytes()).is_err());
    // The level held is a key's, and a map with no key in it holds none:
    // 128 maps read where the innermost is empty. The JSON form counts the
    // map itself, and refuses them.
    let hollow = |depth: usize, open: &str| {
        format!("{}{{}}{}", open.repeat(depth - 1), "}".repeat(depth - 1))
    };
    assert!(read_yaml(hollow(128, "{a: ").as_bytes()).is_ok());
    assert!(read_json(hollow(127, "{\"a\":").as_bytes()).is_ok());
    assert!(read_json(hollow(128, "{\"a\":").as_bytes()).is_err());
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

/// What a read refuses `document` with, in this form.
fn refusal(format: Format, document: &[u8]) -> String {
    format
        .read(document)
        .expect_err("the document is refused")
        .to_string()
}

#[test]
fn a_read_is_the_bound_and_then_the_forms_reader() {
    let document = "apiVersion: smg.policy/v1\nprofiles:\n  default: {pick: random}\n";
    let read = Format::Yaml.read(document.as_bytes()).expect("it reads");
    assert_eq!(Format::Yaml.parse(document.as_bytes()).ok(), Some(read));
    // Past the bound a document is refused by its size, unread: these
    // bytes are no document in either form.
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let over = vec![b'{'; bound + 1];
    for format in [Format::Yaml, Format::Json] {
        assert_eq!(
            refusal(format, &over),
            format!("the document is {} bytes; at most {bound}", bound + 1)
        );
    }
    // At the bound the form's reader has the word, under its form's name.
    let at = &over[..bound];
    assert!(refusal(Format::Yaml, at).starts_with("not a policy spec in yaml: "));
    assert!(refusal(Format::Json, at).starts_with("not a policy spec in json: "));
}

/// A typed YAML read takes each of these, and so a host's own did. This
/// read refuses each by its field or its place, with what a writer does
/// about it where a plain word of YAML is the cause. A null in a list
/// where text, a list or a map stands is an item with nothing in it,
/// and is told so: it has no key to take out.
#[test]
fn a_refusal_names_its_place_and_what_a_writer_does() {
    const QUOTES: &str = "text that reads as a number, a boolean or a null is written in quotes";
    const EMPTY: &str =
        "a key with nothing under it is a null: take the key out, or write [] or {}";
    const NO_TEXT: &str =
        "a key with nothing under it is a null: take the key out, or write its text in quotes";
    const EMPTY_ITEM: &str =
        "a list item with nothing in it is a null: take the item out, or write [] or {}";
    const NO_TEXT_ITEM: &str =
        "a list item with nothing in it is a null: take the item out, or write its text in quotes";
    let document = |profile: &str, more: &str| {
        format!("apiVersion: smg.policy/v1\nprofiles:\n  default: {profile}\n{more}")
    };
    let binding = |model: &str| {
        document(
            "{pick: random}",
            &format!("bindings:\n- match: {{model: {model}}}\n  profile: default\n"),
        )
    };
    for (document, refused) in [
            (
                binding("123"),
                format!("bindings[0].match.model: invalid type: integer `123`, expected a string: {QUOTES}"),
            ),
            (
                binding("true"),
                format!("bindings[0].match.model: invalid type: boolean `true`, expected a string: {QUOTES}"),
            ),
            (
                binding("~"),
                format!("bindings[0].match.model: invalid type: null, expected a string: {NO_TEXT}"),
            ),
            (
                document("{pick: random, score: }", ""),
                format!("profiles.default.score: invalid type: null, expected a sequence: {EMPTY}"),
            ),
            (
                document("{pick: random}", "signals:\n"),
                format!("signals: invalid type: null, expected a map: {EMPTY}"),
            ),
            (
                document("{pick: random}", "view:\n  headers: [~]\n"),
                format!("view.headers[0]: invalid type: null, expected a string: {NO_TEXT_ITEM}"),
            ),
            (
                document("{pick: random}", "view:\n  headers:\n  - x-tenant\n  -\n"),
                format!("view.headers[1]: invalid type: null, expected a string: {NO_TEXT_ITEM}"),
            ),
            (
                document("{pick: random}", "bindings:\n-\n"),
                format!("bindings[0]: invalid type: null, expected struct BindingSpec: {EMPTY_ITEM}"),
            ),
            // A key with nothing under it inside a list's item is a key's.
            (
                document("{pick: random}", "bindings:\n- profiles:\n"),
                format!("bindings[0].profiles: invalid type: null, expected a map: {EMPTY}"),
            ),
            // No cure where neither text, a list nor a map stands.
            (
                document("{pick: random}", "view:\n  boundaries: [~]\n"),
                "view.boundaries[0]: invalid type: null, expected u32".to_owned(),
            ),
            (
                binding("!mine batch"),
                "bindings[0].match.model: the tag `!mine` is none JSON writes: take the tag out at line 5 column 18".to_owned(),
            ),
            // A key's local tag is refused under its map's name, at the place
            // where the map begins. The root has neither: the tag is named.
            (
                document("{!mine pick: random}", ""),
                "profiles.default: the tag `!mine` is none JSON writes: take the tag out at line 3 column 12".to_owned(),
            ),
            (
                document("{pick: random}", "!mine generation: 1\n"),
                "the tag `!mine` is none JSON writes: take the tag out".to_owned(),
            ),
        ] {
            assert_eq!(
                refusal(Format::Yaml, document.as_bytes()),
                format!("not a policy spec in yaml: {refused}"),
                "{document}"
            );
        }
    // A refusal that is no plain word's keeps the reader's own words.
    let unknown = refusal(
        Format::Yaml,
        document("{pick: random}", "more: 1\n").as_bytes(),
    );
    assert!(unknown.starts_with("not a policy spec in yaml: more: unknown field `more`"));
    assert!(!unknown.contains(QUOTES) && !unknown.contains(EMPTY));
    // A JSON document's refusal names its field, and says no cure: the
    // cures are a YAML writer's.
    let json = br#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}},"bindings":[{"match":{"model":123},"profile":"default"}]}"#;
    assert_eq!(
            refusal(Format::Json, json),
            "not a policy spec in json: bindings[0].match.model: invalid type: integer `123`, expected a string"
        );
}

/// A binding's six blocks are optional, and the key of one with nothing
/// under it reads as the key left out: in YAML, where the key stands
/// bare or under `~`, and as a JSON null. The schema file states each
/// of the six as its block or a null. So an emptied `sessions:` keeps
/// no session, and no read says so.
#[test]
fn a_bindings_block_with_nothing_under_it_reads_as_the_key_left_out() {
    const BLOCKS: [&str; 6] = [
        "overload",
        "routing_key",
        "sessions",
        "shadow",
        "canary",
        "admission",
    ];
    // One document, its binding with `more` among its keys, in each form.
    let yaml = |more: &str| {
        format!(
            "apiVersion: smg.policy/v1\nprofiles:\n  default: {{pick: random}}\n\
                 bindings:\n- match: {{model: m}}\n  profile: default\n{more}"
        )
    };
    let json = |more: &str| {
        format!(
                "{{\"apiVersion\":\"smg.policy/v1\",\"profiles\":{{\"default\":{{\"pick\":\"random\"}}}},\
                 \"bindings\":[{{\"match\":{{\"model\":\"m\"}},\"profile\":\"default\"{more}}}]}}"
            )
    };
    // What a document reads as, once its check passed.
    let checked = |format: Format, document: String| {
        let read = format
            .read(document.as_bytes())
            .unwrap_or_else(|refused| panic!("{document}: {refused}"));
        assert!(read.validate().is_ok(), "{document}");
        read
    };
    let left_out = checked(Format::Yaml, yaml(""));
    assert_eq!(checked(Format::Json, json("")), left_out);
    for block in BLOCKS {
        for bare in [format!("  {block}:\n"), format!("  {block}: ~\n")] {
            assert_eq!(checked(Format::Yaml, yaml(&bare)), left_out, "{block}");
        }
        let null = format!(",\"{block}\":null");
        assert_eq!(checked(Format::Json, json(&null)), left_out, "{block}");
    }
}

/// A whole number past 64 bits reads as a fraction where one stands, as
/// the JSON form reads it: both forms read one document. Where a whole
/// number stands, both forms refuse it by its type, in the same words.
#[test]
fn a_whole_number_past_64_bits_reads_as_a_fraction_as_the_json_form_does() {
    let yaml = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    \
                    score: [{least-load: {weight: 18446744073709551616}}]\n    pick: best\n";
    let json = br#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"score":[{"least-load":{"weight":18446744073709551616}}],"pick":"best"}}}"#;
    let read = Format::Yaml
        .read(yaml.as_bytes())
        .expect("the YAML form reads");
    assert_eq!(Format::Json.read(json).ok(), Some(read));

    let refused = "generation: invalid type: floating point `1.8446744073709552e+19`, expected u64";
    let yaml = "apiVersion: smg.policy/v1\ngeneration: 18446744073709551616\n\
                    profiles:\n  default: {pick: random}\n";
    let json = br#"{"apiVersion":"smg.policy/v1","generation":18446744073709551616,"profiles":{"default":{"pick":"random"}}}"#;
    assert_eq!(
        refusal(Format::Yaml, yaml.as_bytes()),
        format!("not a policy spec in yaml: {refused}")
    );
    assert_eq!(
        refusal(Format::Json, json),
        format!("not a policy spec in json: {refused}")
    );
}

/// One document in both forms, with `number` the weight of its one
/// scorer: what the YAML form reads, which the JSON form reads too.
fn weighted(number: &str) -> PolicySpec {
    let yaml = format!(
        "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    \
             score: [{{least-load: {{weight: {number}}}}}]\n    pick: best\n"
    );
    let json = format!(
        r#"{{"apiVersion":"smg.policy/v1","profiles":{{"default":{{"score":[{{"least-load":{{"weight":{number}}}}}],"pick":"best"}}}}}}"#
    );
    let read = Format::Yaml
        .read(yaml.as_bytes())
        .expect("the YAML form reads");
    match Format::Json.read(json.as_bytes()) {
        Ok(json) => assert_eq!(json, read, "the weight {number}"),
        Err(refused) => panic!("the JSON form refuses the weight {number}: {refused}"),
    }
    read
}

/// A number past single precision reads alike in both forms where a
/// single stands: as infinite, which the spec's check refuses by its
/// field. So does the whole number one under the last rounding boundary
/// of single precision: as a double it is the boundary itself. A long
/// number at a rounding boundary reads as one single in both forms too:
/// each form reads a double first, and the field takes it by a cast.
#[test]
fn a_number_past_single_precision_reads_alike_in_both_forms() {
    for past in [
        "340282366920938463463374607431768211455",
        "3.5e38",
        "1e39",
        // 2^128 - 2^103 - 1.
        "340282356779733661637539395458142568447",
    ] {
        let refused = weighted(past)
            .validate()
            .err()
            .map(|refused| refused.to_string());
        assert_eq!(
            refused.as_deref(),
            Some(
                "profiles.default.score[0].least-load.weight = inf is not a finite number \
                     above zero"
            ),
            "{past}"
        );
    }
    let weight = |number: &str| match &weighted(number).profiles["default"].score[0] {
        ScoreSpec::Params(stage) => stage.weight().to_bits(),
        other => panic!("{other:?}"),
    };
    assert_eq!(weight("3.4028235e38"), f32::MAX.to_bits());
    // Just over the middle of the single 1 + 2^-22 and the next above:
    // the nearest double is that middle, and its tie goes to the even
    // single, the lower one.
    assert_eq!(
        weight("1.0000002980232238769531250000000000001"),
        (1.0 + 2.0 * f32::EPSILON).to_bits()
    );
}

/// How many numbers of single precision a schema writes.
fn singles(schema: &Value) -> usize {
    match schema {
        Value::Object(fields) => {
            usize::from(fields.get("format").and_then(Value::as_str) == Some("float"))
                + fields.values().map(singles).sum::<usize>()
        }
        Value::Array(items) => items.iter().map(singles).sum(),
        _ => 0,
    }
}

/// Every field of single precision, at each place a spec writes it,
/// reads a number past it alike in both forms, and the spec's check
/// refuses it by the field's own path: no such field takes an infinite
/// number into a decision. The rows are counted against the schema, so
/// a new field of single precision joins them. The schema writes each
/// field as a `float`, once: three scorers share the bare weight. The
/// overload's usage it writes as a null alone.
#[test]
fn every_field_of_single_precision_refuses_a_number_past_it_in_both_forms() {
    const ABOVE: &str = "a finite number above zero";
    const FROM: &str = "a finite number at or above zero";
    const UNIT: &str = "between 0 and 1";
    const ONE: &str = "a finite number at least 1";
    const STATED: &str = "left out: a worker's registration states this threshold, or its \
                              program's default does";
    let spec = |profile: &str, more: &str| {
        format!(r#"{{"apiVersion":"smg.policy/v1","profiles":{{"default":{profile}}}{more}}}"#)
    };
    let scorer = |name: &str, written: &str| {
        spec(
            &format!(r#"{{"score":[{{"{name}":{written}}}],"pick":"best"}}"#),
            "",
        )
    };
    let mut rows = Vec::new();
    for (name, key, range) in [
        ("least-requests", "weight", ABOVE),
        ("fewest-sessions", "weight", ABOVE),
        ("session-affinity", "weight", ABOVE),
        ("least-load", "weight", ABOVE),
        ("least-load", "kv_pressure_weight", FROM),
        ("least-load", "default_throughput", ABOVE),
        ("prefix-affinity", "weight", ABOVE),
        ("prefix-affinity", "threshold", UNIT),
        ("prefix-affinity", "decay", FROM),
        ("prefix-affinity", "spill.rel", ONE),
        ("kv-pressure", "weight", ABOVE),
        ("kv-pressure", "spread", ABOVE),
        ("kv-pressure", "ceiling", ABOVE),
        ("warmup-slice", "weight", ABOVE),
        ("warmup-slice", "share", UNIT),
        ("warmup-slice", "thin_ratio", UNIT),
        ("warmup-slice", "miss_share", UNIT),
        ("bucket", "weight", ABOVE),
        ("bucket", "rel_threshold", ONE),
        ("consistent-hash", "weight", ABOVE),
    ] {
        // A key under a key: `spill.rel` is written {"spill":{"rel":1e39}}.
        let written = key.rsplit('.').fold("1e39".to_owned(), |inner, key| {
            format!(r#"{{"{key}":{inner}}}"#)
        });
        rows.push((
            scorer(name, &written),
            format!("profiles.default.score[0].{name}.{key} = inf is not {range}"),
        ));
    }
    // Two scorers want a second parameter beside the weight.
    for (name, written) in [
        ("resident-work", r#"{"weight":1e39,"request_tokens":0}"#),
        ("least-cost", r#"{"weight":1e39,"terms":{"load.queued":1}}"#),
    ] {
        rows.push((
            scorer(name, written),
            format!("profiles.default.score[0].{name}.weight = inf is not {ABOVE}"),
        ));
    }
    rows.push((
        spec(
            r#"{"pick":{"best-with-temperature":{"temperature":1e39}}}"#,
            "",
        ),
        format!("profiles.default.pick.best-with-temperature.temperature = inf is not {FROM}"),
    ));
    rows.push((
        spec(
            r#"{"pick":"random"}"#,
            r#","bindings":[{"profile":"default","overload":{"token_usage":1e39}}]"#,
        ),
        format!("bindings[0].overload.token_usage = inf is not {STATED}"),
    ));
    for (document, refused) in &rows {
        // A JSON text is YAML's flow form too: both forms read these
        // bytes.
        let yaml = Format::Yaml
            .read(document.as_bytes())
            .expect("the YAML form reads");
        match Format::Json.read(document.as_bytes()) {
            Ok(json) => assert_eq!(json, yaml, "{document}"),
            Err(refusal) => panic!("the JSON form refuses {document}: {refusal}"),
        }
        let check = yaml.validate().err().map(|refusal| refusal.to_string());
        assert_eq!(check.as_ref(), Some(refused), "{document}");
    }
    let schema = PolicySpec::schema_json().expect("the schema");
    let schema: Value = serde_json::from_str(&schema).expect("the schema is JSON");
    // Two rows more than fields: the three scorers of a bare weight.
    assert_eq!(
        rows.len() - 2,
        singles(&schema) + 1,
        "a field of single precision has no row here"
    );
}

/// A long number reads as the nearest double in both forms, to its last
/// bit: which fraction a whole number past 64 bits reads as is held.
/// 2^64 + 2049 is one past the middle of two doubles, so its lowest bit
/// decides between them.
#[test]
fn a_long_number_reads_as_the_nearest_double_in_both_forms() {
    for (number, nearest) in [
        ("18446744073709553665", 2f64.powi(64) + 4096.0),
        ("-18446744073709553665", -(2f64.powi(64) + 4096.0)),
        ("18446744073710621455", 1.844_674_407_371_062e19),
        ("1.7017420689181831", 1.701_742_068_918_183),
        ("1.9940575818108110", 1.994_057_581_810_811),
    ] {
        let nearest = serde_json::json!({ "a": nearest });
        assert_eq!(
            read_yaml(format!("a: {number}\n").as_bytes()),
            Ok(nearest.clone()),
            "{number}"
        );
        let json = read_json(format!(r#"{{"a":{number}}}"#).as_bytes());
        assert_eq!(
            json.map_err(|refused| refused.to_string()),
            Ok(nearest),
            "{number}"
        );
    }
}

/// A JSON document is read into values first, as a YAML one is. A key
/// written twice is refused by name, a field's as a map's, at the place
/// where its map ends. What stands behind the document's value, white
/// space apart, is refused. And the library reads 127 values deep and no
/// deeper.
#[test]
fn a_json_document_is_read_into_values_first() {
    const SPEC: &str = r#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}}}"#;
    assert!(Format::Json.read(SPEC.as_bytes()).is_ok());
    for (document, refused) in [
        (
            SPEC.replace(
                r#""profiles""#,
                r#""generation":1,"generation":2,"profiles""#,
            ),
            "the key `generation` is written twice at line 1 column 101",
        ),
        (
            SPEC.replace("}}}", r#"},"default":{"pick":"random"}}}"#),
            "the key `default` is written twice at line 1 column 98",
        ),
        (
            SPEC.replace(
                r#"{"pick":"random"}"#,
                r#"{"score":[{"least-load":{"weight":1,"weight":2}}],"pick":"best"}"#,
            ),
            "the key `weight` is written twice at line 1 column 99",
        ),
        (
            format!("{SPEC} {{}}"),
            "trailing characters at line 1 column 73",
        ),
    ] {
        assert_eq!(
            refusal(Format::Json, document.as_bytes()),
            format!("not a policy spec in json: {refused}"),
            "{document}"
        );
    }
    let nest = |depth: usize| format!("{}{}", "[".repeat(depth), "]".repeat(depth));
    assert!(read_json(nest(127).as_bytes()).is_ok());
    assert_eq!(
        read_json(nest(128).as_bytes()).map_err(|refused| refused.to_string()),
        Err("recursion limit exceeded at line 1 column 128".to_owned())
    );
}

/// A JSON document has no alias, so one of the size bound is under both
/// bounds of the read. These are its densest forms: a list of one-digit
/// numbers by the values it builds, one value for each two bytes; one
/// text by the text it reads as; and keys of a number each. A key
/// written twice to the size bound is weighed whole too, and refused by
/// its name, not by a bound. The read is held to its two bounds all the
/// same: a text past twice the size bound, handed to it past the size
/// bound, is refused by the text's.
#[test]
fn a_json_document_of_the_size_bound_is_under_both_bounds_of_the_read() {
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let numbers = format!("[{}0]", "0,".repeat((bound - 3) / 2));
    assert_eq!(numbers.len(), bound - 1, "a byte under the bound");
    let read = read_json(numbers.as_bytes()).expect("the numbers read");
    assert_eq!(read.as_array().map(Vec::len), Some(bound / 2 - 1));

    let text = format!("\"{}\"", "x".repeat(bound - 2));
    let read = read_json(text.as_bytes()).expect("the text reads");
    assert_eq!(read.as_str().map(str::len), Some(bound - 2));

    let mut keys = String::from("{\"0\":0");
    let mut key = 1;
    while keys.len() + 12 < bound {
        let _ = write!(keys, ",\"{key:x}\":0");
        key += 1;
    }
    keys.push('}');
    let read = read_json(keys.as_bytes()).expect("the keys read");
    assert_eq!(read.as_object().map(Map::len), Some(key));

    let twice = format!("{{{}\"\":0}}", "\"\":0,".repeat((bound - 7) / 5));
    assert!(twice.len() > bound - 5 && twice.len() <= bound);
    assert_eq!(
        read_json(twice.as_bytes()).map_err(|refused| refused.to_string()),
        Err(format!(
            "the key `` is written twice at line 1 column {}",
            twice.len()
        ))
    );

    let past = format!("\"{}\"", "x".repeat(2 * bound + 1));
    match read_json(past.as_bytes()) {
        Err(refused) => assert!(refused.to_string().contains(PAST_TEXT), "{refused}"),
        Ok(_) => panic!("{} bytes of text read", past.len() - 2),
    }
}

/// The three numbers the two forms read apart, each by its form's own
/// rule: a zero written `-0`, a number past a double, and a whole number
/// in hex, octal or binary.
#[test]
fn the_two_forms_read_three_texts_apart() {
    let read = |format: Format, generation: &str| {
        let document = match format {
            Format::Yaml => format!(
                "apiVersion: smg.policy/v1\ngeneration: {generation}\n\
                     profiles:\n  default: {{pick: random}}\n"
            ),
            Format::Json => format!(
                r#"{{"apiVersion":"smg.policy/v1","generation":{generation},"profiles":{{"default":{{"pick":"random"}}}}}}"#
            ),
        };
        format
            .read(document.as_bytes())
            .map_err(|refused| refused.to_string())
    };
    // `-0` is the whole number zero in YAML, and a fraction in JSON.
    assert_eq!(read_yaml(b"a: -0\n"), Ok(serde_json::json!({"a": 0})));
    let fraction = read_json(br#"{"a":-0}"#).expect("it reads");
    assert!(fraction["a"].is_f64());
    assert!(fraction["a"].as_f64().is_some_and(f64::is_sign_negative));
    let zero = read(Format::Yaml, "0");
    assert!(zero.is_ok(), "{zero:?}");
    assert_eq!(read(Format::Yaml, "-0"), zero);
    assert_eq!(
        read(Format::Json, "-0"),
        Err(
            "not a policy spec in json: generation: invalid type: floating point `-0.0`, \
                 expected u64"
                .to_owned()
        )
    );
    // A number past a double is text in YAML, and out of range in JSON.
    assert_eq!(
        read(Format::Yaml, "1e400"),
        Err(
            "not a policy spec in yaml: generation: invalid type: string \"1e400\", \
                 expected u64"
                .to_owned()
        )
    );
    assert_eq!(
        read(Format::Json, "1e400"),
        Err("not a policy spec in json: number out of range at line 1 column 48".to_owned())
    );
    // A YAML whole number in hex, octal or binary: JSON has no such form.
    for sixteen in ["0x10", "0o20", "0b10000"] {
        assert_eq!(
            read(Format::Yaml, sixteen),
            read(Format::Yaml, "16"),
            "{sixteen}"
        );
        assert_eq!(
            read(Format::Json, sixteen),
            Err("not a policy spec in json: expected `,` or `}` at line 1 column 45".to_owned()),
            "{sixteen}"
        );
    }
    assert!(read(Format::Yaml, "16").is_ok() && read(Format::Json, "16").is_ok());
}

/// The crate's manifest asks the JSON library for its exact reading
/// (`float_roundtrip`) in its own entry of that library. No case of
/// behaviour can hold that line: every test build of the crate has the
/// feature on without it, by the ask of another crate that the cases'
/// build reaches, so a case reads a long number the exact way with the
/// line and without it. The crate built alone, and a package built on
/// it that reaches no such crate, have the feature by this line only.
/// So this case reads the entry itself.
#[test]
fn the_crates_manifest_asks_the_json_library_for_its_exact_reading() {
    let manifest = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
    let dependencies = manifest
        .split_once("\n[dependencies]\n")
        .map(|(_, rest)| rest.split("\n[").next().unwrap_or(rest))
        .expect("the manifest's dependencies");
    let entries: Vec<&str> = dependencies
        .lines()
        .filter(|line| line.starts_with("serde_json = "))
        .collect();
    let [entry] = entries.as_slice() else {
        panic!("one entry of the JSON library: {entries:?}")
    };
    // A comment behind the entry asks for nothing.
    let written = entry.split('#').next().unwrap_or_default();
    let features = written
        .split_once("features = [")
        .and_then(|(_, rest)| rest.split_once(']'))
        .map_or("", |(features, _)| features);
    assert!(
        features
            .split(',')
            .any(|feature| feature.trim() == "\"float_roundtrip\""),
        "{entry}"
    );
}
