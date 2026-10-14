//! Rows recorded from Python in `tests/fixtures/hf/`: the regex reader and
//! matcher against the `regex` module (`regex*.jsonl`), and the Python
//! built-ins and character tables (`units.jsonl`).

use serde_json::{json, Map, Value};

use crate::{
    content_parsers::{Content, ContentParser},
    error::{PyErrorKind, Taint, Unsupported},
    load_response_template, py,
    pyre::{
        self,
        parse::{ReadError, Regex},
        Found, Match, Memo, Pattern, Role,
    },
    response_parser::{coerce, schema_types},
    test_fixtures::{
        byte_offset, canonical, char_index, digest_text, fixtures_dir, read_jsonl, tag, untag,
    },
    ResponseParser,
};

fn span(hay: &str, start: usize, end: usize) -> String {
    format!("{}:{}", char_index(hay, start), char_index(hay, end))
}

fn groups(pattern: &Pattern, hay: &str, m: &Match) -> Value {
    let map: Map<String, Value> = pattern
        .groupdict(hay, m)
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value.map_or(Value::Null, Value::from)))
        .collect();
    Value::Object(map)
}

/// The recorded results of one regex row, computed with the port.
fn port_results(pattern: &Pattern, row: &Value) -> Value {
    let mut results = Vec::new();
    for result in row["results"].as_array().unwrap() {
        let hay = result["hay"].as_str().unwrap();
        if row["role"] == "finditer" {
            let found: Vec<Value> = pattern
                .finditer(hay)
                .map(|m| json!([span(hay, m.start, m.end), groups(pattern, hay, &m)]))
                .collect();
            results.push(json!({"hay": hay, "finditer": found}));
            continue;
        }
        let mut searches = Vec::new();
        for entry in result["search"].as_array().unwrap() {
            let pos = byte_offset(hay, entry["pos"].as_u64().unwrap() as usize);
            let mut out = json!({"pos": entry["pos"]});
            if let Some(m) = pattern.search(hay, pos, &mut Memo::default()) {
                out["full"] = span(hay, m.start, m.end).into();
                out["groups"] = groups(pattern, hay, &m);
            }
            match pattern.search_partial(hay, pos, &mut Memo::default()) {
                Some(Found::Complete(m)) if m.start < hay.len() => {
                    out["partial"] = span(hay, m.start, m.end).into();
                }
                Some(Found::Partial(start)) => {
                    out["partial"] = format!("{}+", span(hay, start, hay.len())).into();
                }
                _ => {}
            }
            searches.push(out);
        }
        results.push(json!({"hay": hay, "search": searches}));
    }
    Value::Array(results)
}

/// The reader reads every pattern the `regex` module compiles, except a
/// construct it does not read (`NotRead`, listed in the README), and rejects
/// every one the module rejects. The matcher returns what the module returns
/// for every pattern it runs, and refuses the others.
#[test]
fn regex_rows_match_the_regex_module() {
    let rows = read_jsonl(&fixtures_dir(), "regex");
    assert!(!rows.is_empty(), "no regex rows");
    let (mut compared, mut refused, mut not_read) = (0, 0, 0);
    let mut failures = Vec::new();
    for row in &rows {
        let source = row["pattern"].as_str().unwrap();
        let role = if row["role"] == "finditer" {
            Role::Finditer
        } else {
            Role::Delimiter
        };
        match (row["python"] == "ok", Regex::parse(source)) {
            (true, Err(ReadError::Syntax(why))) => {
                failures.push(format!(
                    "{source:?}: the regex module reads it, the reader rejects it: {why}"
                ));
            }
            (true, Err(ReadError::NotRead(_))) => not_read += 1,
            (false, Ok(_)) => failures.push(format!(
                "{source:?}: the regex module rejects it, the reader reads it"
            )),
            _ => {}
        }
        match (row["python"] == "ok", Pattern::new(source, role)) {
            (true, Ok(pattern)) => {
                compared += 1;
                let names: Vec<&str> = pattern.group_names().collect();
                if json!(names) != row["groups"] {
                    failures.push(format!("{source:?}: groups {names:?}"));
                }
                let got = port_results(&pattern, row);
                if got != row["results"] {
                    failures.push(format!(
                        "{source:?} ({}): got {got}\nexpected {}",
                        row["role"], row["results"]
                    ));
                }
            }
            (true, Err(what)) if row.get("corpus").is_some() => {
                failures.push(format!("{source:?}: a corpus pattern is refused: {what}"));
            }
            (true, Err(_)) => refused += 1,
            (false, Ok(_)) => failures.push(format!(
                "{source:?}: Python rejects it, the port compiles it"
            )),
            (false, Err(_)) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} regex rows differ ({compared} compared, {refused} refused, {not_read} not read):\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

/// The `regex` module drops an empty group, also with `?`, before it merges
/// alternatives of one character, so each of these matches as `[^ab]`: it
/// finds nothing in "a" or "b", where Python's `re` finds both (`regex`
/// 2026.9.10, transformers 5.18.0). The port refuses them or matches alike.
#[test]
fn empty_groups_do_not_hide_merged_negated_classes() {
    for source in [
        "[^a](?:)|[^b]",
        "[^a](?:)?|[^b]",
        "[^a](?:(?:))|[^b]",
        "(?:[^a](?:))|[^b]",
        "[^a](?:)(?:)|[^b]",
        "[^a]|[^b](?:)",
    ] {
        for role in [Role::Delimiter, Role::Finditer] {
            let Ok(pattern) = Pattern::new(source, role) else {
                continue;
            };
            for hay in ["a", "b"] {
                assert!(
                    pattern.search(hay, 0, &mut Memo::default()).is_none(),
                    "{source:?} ({role:?}) finds {hay:?}; the regex module does not"
                );
            }
        }
    }
    // transformers does not close the region, since "a>" does not match.
    let spec = json!({"start_anchor": "<s>", "fields": {
        "thinking": {"open": "<t>", "close_pattern": "(?:[^a](?:)|[^b])>"},
        "content": {},
    }});
    if let Ok(template) = load_response_template(&spec) {
        let message = crate::parse_response("<t>xyz a>rest", &template, "", &[]).unwrap();
        assert_eq!(json!(message), json!({"thinking": "xyz a>rest"}));
    }
}

/// `\w` and `\s` in a class, with what the `regex` module returns (`regex`
/// 2026.9.10, `regex.compile(pattern, regex.DOTALL)`; spans in characters).
/// Its `\s` is Unicode White_Space: U+001C, which Python's `re` counts as
/// whitespace, is not one. Alternatives with such a negated class are not
/// merged the way `[^a]|[^b]` is. A class with a class escape and its
/// complement matches every character, also when it is negated: the module
/// makes it any character and drops the negation.
#[test]
fn class_escapes_match_the_regex_module() {
    let hay = "a\u{1c}b\u{a0}c\u{2028}d\u{3000}e\u{85}f\tg>h é_1-";
    let spaces: &[&str] = &["3:4", "5:6", "7:8", "9:10", "11:12", "15:16"];
    let every: Vec<String> = (0..20).map(|k| format!("{k}:{}", k + 1)).collect();
    let every: Vec<&str> = every.iter().map(String::as_str).collect();
    for (source, expected) in [
        (r"[\s]", spaces),
        (
            r"[^\s>]+",
            &[
                "0:3", "4:5", "6:7", "8:9", "10:11", "12:13", "14:15", "16:20",
            ],
        ),
        (
            r"[\w-]+",
            &[
                "0:1", "2:3", "4:5", "6:7", "8:9", "10:11", "12:13", "14:15", "16:20",
            ],
        ),
        (r"[^\w\s]", &["1:2", "13:14", "19:20"]),
        (
            r"[\s-]+",
            &["3:4", "5:6", "7:8", "9:10", "11:12", "15:16", "19:20"],
        ),
        (r"[^\s]|[^a]", &every),
        (r"x?[^a\s]|[^b]", &every),
        (r"[^\s\S]", &every),
        (r"[^a\d\D]", &every),
    ] {
        for role in [Role::Delimiter, Role::Finditer] {
            let pattern = Pattern::new(source, role).unwrap();
            let got: Vec<String> = pattern
                .finditer(hay)
                .map(|m| span(hay, m.start, m.end))
                .collect();
            assert_eq!(got, expected, "{source:?} ({role:?})");
        }
    }

    // An opener that names a call: `search` (span and name) and
    // `search(..., partial=True)` from 0; a partial match ending in "+" can
    // still grow.
    let opener = Pattern::new(r"<call:(?P<name>[^\s>]+)>", Role::Delimiter).unwrap();
    for (hay, full, partial) in [
        (
            "<call:get_weather>",
            Some(("0:18", "get_weather")),
            Some("0:18"),
        ),
        (
            "<call:ns.tool-x>",
            Some(("0:16", "ns.tool-x")),
            Some("0:16"),
        ),
        ("<call:a\u{1c}b>", Some(("0:10", "a\u{1c}b")), Some("0:10")),
        ("<call:a<b>", Some(("0:10", "a<b")), Some("0:10")),
        ("<call:a\u{a0}b>", None, None),
        ("<call:a\u{2028}b>", None, None),
        ("<call:a b>", None, None),
        ("<call:get_we", None, Some("0:12+")),
        ("<call:a", None, Some("0:7+")),
    ] {
        let got = opener
            .search(hay, 0, &mut Memo::default())
            .map(|m| (span(hay, m.start, m.end), opener.groupdict(hay, &m)));
        let expected = full.map(|(at, name)| (at.to_owned(), vec![("name", Some(name))]));
        assert_eq!(got, expected, "{hay:?}");
        let got = match opener.search_partial(hay, 0, &mut Memo::default()) {
            Some(Found::Complete(m)) => Some(span(hay, m.start, m.end)),
            Some(Found::Partial(start)) => Some(format!("{}+", span(hay, start, hay.len()))),
            None => None,
        };
        assert_eq!(got.as_deref(), partial, "{hay:?}");
    }
}

/// Lookahead in a pattern used only with `finditer`, such as a tag pattern
/// whose closing tag may be left out, with what the `regex`
/// module returns (`regex` 2026.9.10, spans in characters). Lookahead in an
/// open or close pattern, and a group, backreference or repeat in or of one,
/// stay refused.
#[test]
fn lookahead_in_finditer_patterns() {
    type Spans<'a> = (String, Vec<(&'a str, Option<&'a str>)>);
    let rows = read_jsonl(&fixtures_dir(), "regex");
    let tag = r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*(?:</parameter>|(?=<parameter=|$))";
    for source in [tag, r"a(?=b(?=c))"] {
        assert!(
            rows.iter()
                .any(|r| r["pattern"] == source && r["role"] == "finditer" && r["python"] == "ok"),
            "{source:?} is not recorded"
        );
    }
    let pattern = Pattern::new(tag, Role::Finditer).unwrap();
    for (hay, expected) in [
        (
            "<parameter=a>\n1\n<parameter=b>\n2\n</parameter>",
            &[("0:16", "a", "1"), ("16:44", "b", "2")][..],
        ),
        (
            "<parameter=a>\nx\n<parameter=b\n</parameter>",
            &[("0:16", "a", "x")][..],
        ),
        (
            "<parameter=a>\n1\n</parameter><parameter=b>\n2\n",
            &[("0:28", "a", "1"), ("28:44", "b", "2")][..],
        ),
    ] {
        let got: Vec<Spans<'_>> = pattern
            .finditer(hay)
            .map(|m| (span(hay, m.start, m.end), pattern.groupdict(hay, &m)))
            .collect();
        let expected: Vec<Spans<'_>> = expected
            .iter()
            .map(|&(at, key, value)| {
                (
                    at.to_owned(),
                    vec![("key", Some(key)), ("value", Some(value))],
                )
            })
            .collect();
        assert_eq!(got, expected, "{hay:?}");
    }
    let nested = Pattern::new(r"a(?=b(?=c))", Role::Finditer).unwrap();
    let hay = "abcab abc";
    let got: Vec<String> = nested
        .finditer(hay)
        .map(|m| span(hay, m.start, m.end))
        .collect();
    assert_eq!(got, ["0:1", "6:7"]);
    for (source, role, reason) in [
        ("a(?=b)", Role::Delimiter, "regex: look-around"),
        ("a(?!b)", Role::Finditer, "regex: look-around"),
        ("(?<=a)b", Role::Finditer, "regex: look-around"),
        (
            "a(?=(?P<x>b))",
            Role::Finditer,
            "regex: group in a lookahead",
        ),
        (
            r"(?P<k>a)(?=\1)",
            Role::Finditer,
            "regex: backreference in a lookahead",
        ),
        (
            "a(?=b)?",
            Role::Finditer,
            "regex: repeat of a group or an assertion",
        ),
        ("(?=a)", Role::Finditer, "empty match"),
    ] {
        let refused = Pattern::new(source, role).unwrap_err().to_string();
        assert_eq!(refused, reason, "{source:?}");
    }
}

/// Class escapes in classes, with what the `regex` module returns (`regex`
/// 2026.9.10, `regex.compile(pattern, regex.DOTALL)`; spans in characters).
/// The module reads a `-` next to a class escape as a member (`[\s-a]` is
/// `\s`, `-` and `a`), where Python's `re` rejects the range, and `\b` in a
/// class as a backspace. Two negated classes of one character stay refused
/// next to one with `\s`.
#[test]
fn class_escapes_in_classes() {
    let hay = "a b-\u{8}c1\u{661}\u{e9}_>\t\u{3000}x";
    for (source, expected) in [
        (r"[\s-a]", &["0:1", "1:2", "3:4", "11:12", "12:13"][..]),
        (
            r"[\w--\d]",
            &[
                "0:1", "2:3", "3:4", "5:6", "6:7", "7:8", "8:9", "9:10", "13:14",
            ],
        ),
        (r"[a-\s]", &["0:1", "1:2", "3:4", "11:12", "12:13"]),
        (r"[\d\s]", &["1:2", "6:7", "7:8", "11:12", "12:13"]),
        (r"[^\S]", &["1:2", "11:12", "12:13"]),
        (r"[\W\s]", &["1:2", "3:4", "4:5", "10:11", "11:12", "12:13"]),
        (r"[\s\b]", &["1:2", "4:5", "11:12", "12:13"]),
        (r"[^\d\W]+", &["0:1", "2:3", "5:6", "8:10", "13:14"]),
        (r"\S+", &["0:1", "2:11", "13:14"]),
        (r"\W+", &["1:2", "3:5", "10:13"]),
        (r"\d+", &["6:8"]),
    ] {
        for role in [Role::Delimiter, Role::Finditer] {
            let pattern = Pattern::new(source, role).unwrap();
            let got: Vec<String> = pattern
                .finditer(hay)
                .map(|m| span(hay, m.start, m.end))
                .collect();
            assert_eq!(got, expected, "{source:?} ({role:?})");
        }
    }
    for role in [Role::Delimiter, Role::Finditer] {
        match Pattern::new(r"[^\s]|[^a]|[^b]", role) {
            Err(Unsupported::Regex(got)) => assert_eq!(
                got, "alternatives that can end in a negated class of one character",
                "{role:?}"
            ),
            other => panic!("{role:?}: {other:?}"),
        }
    }
}

/// Groups nest at most 100 deep; deeper patterns are refused before parsing
/// recurses further (the `regex` module takes up to 198 levels). A template
/// is read on a request's tokio worker, whose stack is 2 MiB.
#[test]
fn deeply_nested_groups_are_refused() {
    let nested = |depth: usize, closed: bool| {
        let close = if closed {
            ")".repeat(depth)
        } else {
            String::new()
        };
        format!("{}a{close}", "(?:".repeat(depth))
    };
    let worker = std::thread::Builder::new().stack_size(2 << 20);
    let run = move || {
        for role in [Role::Delimiter, Role::Finditer] {
            assert!(Pattern::new(&nested(100, true), role).is_ok());
            for (depth, closed) in [(101, true), (3_000, true), (3_000, false), (100_000, true)] {
                match Pattern::new(&nested(depth, closed), role) {
                    Err(Unsupported::Regex(reason)) => {
                        assert_eq!(reason, "groups nested more than 100 deep", "{depth}");
                    }
                    other => panic!("{depth} {closed}: {other:?}"),
                }
            }
        }
        let template = |depth| {
            json!({"start_anchor": "<s>", "fields": {
                "x": {"open_pattern": nested(depth, true), "close": "</x>"}, "content": {}}})
        };
        assert!(load_response_template(&template(100)).is_ok());
        assert!(load_response_template(&template(3_000)).is_err());
    };
    worker.spawn(run).unwrap().join().unwrap();
}

/// Parse errors name the field and what failed, never the model output:
/// callers log them.
#[test]
fn parse_errors_do_not_quote_the_output() {
    let spec = json!({"start_anchor": "<s>", "fields": {
        "tool_calls": {"open": "<call>", "close": "</call>", "content": "json"},
        "content": {}}});
    let template = load_response_template(&spec).unwrap();
    let region = "{\"card\": \"4111 1111 1111 1111\",\n oops}";
    let output = format!("<call>{region}</call>");
    let error = crate::parse_response(&output, &template, "", &[]).unwrap_err();
    let expected = format!(
        "field 'tool_calls': ValueError: json parser could not parse region as JSON ({} bytes)",
        region.len()
    );
    assert_eq!(error.to_string(), expected);
}

/// Characters the `regex` module reads as literals where they start nothing
/// else (`{`, `}`, `]`, and `|` or `^` inside a class), counted repeats, and
/// escapes of control characters and code points, with what it returns
/// (`regex` 2026.9.10, spans in characters). A quantifier with nothing to
/// repeat is rejected as the module rejects it.
#[test]
fn braces_brackets_and_escapes() {
    let hay = "aa{2}a}a]x|^y\tz a{x} {} a\tq";
    for (source, expected) in [
        ("a{2}", &["0:2"][..]),
        ("a{x}", &["16:20"]),
        ("a}", &["5:7"]),
        ("a]", &["7:9"]),
        (
            "[]a]",
            &["0:1", "1:2", "5:6", "7:8", "8:9", "16:17", "24:25"],
        ),
        (
            "[a|b]",
            &["0:1", "1:2", "5:6", "7:8", "10:11", "16:17", "24:25"],
        ),
        (
            "[a^b]",
            &["0:1", "1:2", "5:6", "7:8", "11:12", "16:17", "24:25"],
        ),
        ("[|^]", &["10:11", "11:12"]),
        (r"a\t", &["24:26"]),
        (r"\141\u0009", &["24:26"]),
        (r"\x61{1,2}", &["0:2", "5:6", "7:8", "16:17", "24:25"]),
        (r"a{,1}\x7b", &["1:3", "16:18", "21:22"]),
        (r"\{\}", &["21:23"]),
        ("{}", &["21:23"]),
    ] {
        for role in [Role::Delimiter, Role::Finditer] {
            let pattern = Pattern::new(source, role).unwrap();
            let got: Vec<String> = pattern
                .finditer(hay)
                .map(|m| span(hay, m.start, m.end))
                .collect();
            assert_eq!(got, expected, "{source:?} ({role:?})");
        }
    }
    for (source, reason) in [
        ("*a", "nothing to repeat"),
        ("a**", "multiple repeat"),
        ("a{2,1}", "min repeat greater than max repeat"),
        ("(?P<n>a)(?P=m)", "unknown group"),
    ] {
        assert_eq!(
            Regex::parse(source).err(),
            Some(ReadError::Syntax(reason.into())),
            "{source:?}"
        );
    }
}

#[test]
fn the_last_start_anchor_ends_where_the_tail_begins() {
    let spec = json!({"start_anchor": "<a>", "fields": {"content": {}}});
    let template = load_response_template(&spec).unwrap();
    assert_eq!(template.last_anchor_end("x<a>y<a>z"), Some(8));
    assert_eq!(template.truncate_past_last_anchor("x<a>y<a>z"), "z");
    assert_eq!(template.last_anchor_end("x<a>"), Some(4));
    // Without an anchor transformers keeps the whole text.
    assert_eq!(template.last_anchor_end("xyz"), None);
    assert_eq!(template.truncate_past_last_anchor("xyz"), "xyz");
}

fn class(kind: PyErrorKind) -> Value {
    Value::from(kind.to_string())
}

/// `{"output": tagged}` or `{"error": class}`, as the generator records it.
fn outcome(result: Result<Value, Value>, taint: Taint) -> Value {
    match (result, taint) {
        (Err(class), _) => json!({"error": class}),
        (Ok(_), Some(_)) => json!({"error": "Unrepresentable"}),
        (Ok(value), None) => {
            let tagged = tag(&value);
            if depth(&tagged) > 64 {
                json!({"output_digest": digest_text(&canonical(&tagged))})
            } else {
                json!({"output": tagged})
            }
        }
    }
}

fn depth(v: &Value) -> usize {
    match v {
        Value::Array(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
        Value::Object(map) => 1 + map.values().map(depth).max().unwrap_or(0),
        _ => 0,
    }
}

fn recorded(row: &Value) -> Value {
    let mut out = Map::new();
    for key in ["output", "error", "output_digest"] {
        if let Some(v) = row.get(key) {
            out.insert(key.to_owned(), v.clone());
        }
    }
    Value::Object(out)
}

/// Python's result for a unit row, computed with the port; `None` for rows
/// the port does not reproduce (refused content_args, a Python `None` text).
fn port_unit(row: &Value) -> Option<Value> {
    let args = row.get("args").map(untag);
    Some(match row["kind"].as_str().unwrap() {
        "int" => match py::int(row["input"].as_str().unwrap()) {
            Some((value, taint)) => outcome(Ok(value), taint),
            None => outcome(Err("ValueError".into()), None),
        },
        "float" => match py::float(row["input"].as_str().unwrap()) {
            Some(f) => {
                let (value, taint) = py::float_value(f);
                outcome(Ok(value), taint)
            }
            None => outcome(Err("ValueError".into()), None),
        },
        "strip" => json!({"output": py::strip(row["input"].as_str().unwrap())}),
        "json_loads" => {
            let mut taint = None;
            let result = py::json_loads(row["input"].as_str().unwrap(), &mut taint).map_err(|e| {
                Value::from(match e {
                    py::JsonError::Recursion => "RecursionError",
                    _ => "ValueError",
                })
            });
            outcome(result, taint)
        }
        "coerce" => {
            let args = args.unwrap();
            let types: Vec<String> = serde_json::from_value(args[1].clone()).unwrap();
            let mut taint = None;
            let result =
                coerce(args[0].as_str().unwrap(), &types, &mut taint).map_err(|e| class(e.0));
            outcome(result, taint)
        }
        "schema_types" => {
            let result = schema_types(&args.unwrap()[0])
                .map(|types| json!(types))
                .map_err(|e| class(e.0));
            outcome(result, None)
        }
        "parse_content" => {
            let args = args.unwrap();
            let text = args[0].as_str()?;
            let parser = ContentParser::from_name(args[1].as_str()?)?;
            let content = Content::new(parser, Some(&args[2])).ok()?;
            let mut taint = None;
            let result = content.parse(text, &mut taint).map_err(|e| class(e.0));
            outcome(result, taint)
        }
        "coerce_tool_calls" => {
            let spec = json!({"start_anchor": "a", "fields": {"b": {}}});
            let template = load_response_template(&spec).unwrap();
            let tools: Vec<Value> = untag(&row["tools"]).as_array().cloned().unwrap_or_default();
            let parser = ResponseParser::new(&template, "", &tools).unwrap();
            let mut value = untag(&row["value"]);
            let mut taint = None;
            let result = parser
                .coerce_tool_calls(&mut value, &mut taint)
                .map(|()| value);
            outcome(result.map_err(|e| class(e.0)), taint)
        }
        _ => return None,
    })
}

#[test]
fn units_match_python() {
    let rows = read_jsonl(&fixtures_dir(), "units");
    let mut failures = Vec::new();
    let (mut compared, mut skipped) = (0, 0);
    for row in rows.iter().filter(|r| r["kind"] != "table") {
        let Some(got) = port_unit(row) else {
            skipped += 1;
            continue;
        };
        compared += 1;
        let expected = recorded(row);
        if got != expected {
            failures.push(format!(
                "{}: got {got}, expected {expected}",
                canonical(row)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {compared} unit rows differ ({skipped} skipped):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn character_tables_match_python() {
    let rows = read_jsonl(&fixtures_dir(), "units");
    let tables: Vec<&Value> = rows.iter().filter(|r| r["kind"] == "table").collect();
    assert_eq!(tables.len(), 5, "missing table rows");
    for table in tables {
        let check: fn(char) -> bool = match table["name"].as_str().unwrap() {
            "regex_word" => pyre::is_word,
            "regex_space" => char::is_whitespace,
            "regex_digit" => pyre::is_digit,
            "str_isspace" => py::is_space,
            "str_isdecimal" => |c| py::int(&c.to_string()).is_some(),
            other => panic!("unknown table {other}"),
        };
        let mut inside = vec![false; 0x11_0000];
        for range in table["ranges"].as_array().unwrap() {
            let lo = range[0].as_u64().unwrap() as usize;
            let hi = range[1].as_u64().unwrap() as usize;
            inside[lo..=hi].iter_mut().for_each(|x| *x = true);
        }
        let differing: Vec<String> = (0..0x11_0000u32)
            .filter_map(char::from_u32)
            .filter(|&c| check(c) != inside[c as usize])
            .map(|c| format!("U+{:04X}", c as u32))
            .collect();
        assert!(
            differing.is_empty(),
            "{}: {} code points differ: {:?}",
            table["name"],
            differing.len(),
            &differing[..differing.len().min(20)]
        );
    }
}
