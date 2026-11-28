use std::collections::HashMap;

use super::*;
use crate::RlError;

fn view(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn parses_and_matches_each_operator() {
    let v = view(&[("engine", "sglang"), ("tp_size", "2"), ("role", "policy")]);
    let cases: &[(&str, bool)] = &[
        ("engine=sglang", true),
        ("engine=vllm", false),
        ("engine!=vllm", true),
        ("engine!=sglang", false),
        ("tp_size in (1,2,4)", true),
        ("tp_size in (8)", false),
        ("role notin (reward, judge)", true),
        ("role notin (policy)", false),
        ("engine=sglang, tp_size=2", true),
        ("engine=sglang,tp_size=4", false),
        ("  engine = sglang ,  role = policy  ", true),
        ("engine in(sglang,vllm)", true),
        ("missing=x", false),
        ("missing!=x", true),
        ("missing in (x)", false),
        ("missing notin (x)", true),
        (r#"role="policy""#, true),
        (r#"role="pol\"icy""#, false),
    ];
    for (expr, expected) in cases {
        let sel = Selector::parse(expr).unwrap_or_else(|e| panic!("{expr}: {e}"));
        assert_eq!(sel.matches(&v), *expected, "{expr}");
    }
}

#[test]
fn quoted_values_allow_commas_and_escapes() {
    let v = view(&[("model_path", "/models/a,b"), ("q", r#"say "hi""#)]);
    assert!(Selector::parse(r#"model_path="/models/a,b""#)
        .unwrap()
        .matches(&v));
    assert!(Selector::parse(r#"q="say \"hi\"""#).unwrap().matches(&v));
}

#[test]
fn reports_parse_errors_with_offsets() {
    let cases: &[(&str, usize)] = &[
        ("", 0),
        ("=x", 0),
        ("engine", 6),
        ("engine==x", 7),
        ("engine in x", 10),
        ("engine in (a,", 13),
        ("engine=a b", 9),
        ("engine=\"unterminated", 20),
        ("engine=a,", 9),
        ("engine=", 7),
    ];
    for (expr, offset) in cases {
        match Selector::parse(expr) {
            Err(RlError::InvalidSelector { offset: got, .. }) => {
                assert_eq!(got, *offset, "{expr:?}");
            }
            other => panic!("{expr:?}: expected InvalidSelector, got {other:?}"),
        }
    }
}

#[test]
fn keeps_source_text() {
    let sel = Selector::parse("engine=sglang").unwrap();
    assert_eq!(sel.source(), "engine=sglang");
}
