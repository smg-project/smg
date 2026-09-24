//! Label selector: `k=v`, `k!=v`, `k in (a,b)`, `k notin (a,b)`, comma = AND.

use std::collections::HashMap;

use crate::error::RlError;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    Eq,
    Ne,
    In,
    NotIn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Term {
    key: String,
    op: Op,
    values: Vec<String>,
}

/// A parsed selector. Terms are AND-ed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selector {
    source: String,
    terms: Vec<Term>,
}

struct Scanner<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn err<T>(&self, message: &str) -> Result<T, RlError> {
        Err(RlError::InvalidSelector {
            offset: self.pos,
            message: message.to_string(),
        })
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.bump();
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.src[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }

    fn key(&mut self) -> Result<String, RlError> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '/' | '-'))
        {
            self.bump();
        }
        if self.pos == start {
            return self.err("expected key");
        }
        Ok(self.src[start..self.pos].to_string())
    }

    fn op(&mut self) -> Result<Op, RlError> {
        if self.eat("!=") {
            return Ok(Op::Ne);
        }
        if self.eat("=") {
            if self.peek() == Some('=') {
                return self.err("expected value after `=`");
            }
            return Ok(Op::Eq);
        }
        if self.word_op("notin") {
            return Ok(Op::NotIn);
        }
        if self.word_op("in") {
            return Ok(Op::In);
        }
        self.err("expected operator (`=`, `!=`, `in`, `notin`)")
    }

    /// `in` / `notin` must be followed by whitespace or `(` to count as an operator.
    fn word_op(&mut self, word: &str) -> bool {
        let rest = &self.src[self.pos..];
        if !rest.starts_with(word) {
            return false;
        }
        let after = rest[word.len()..].chars().next();
        if matches!(after, Some('(')) || after.is_some_and(char::is_whitespace) {
            self.pos += word.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<String, RlError> {
        if self.peek() == Some('"') {
            return self.quoted();
        }
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| !c.is_whitespace() && !matches!(c, ',' | '(' | ')' | '"'))
        {
            self.bump();
        }
        if self.pos == start {
            return self.err("expected value");
        }
        Ok(self.src[start..self.pos].to_string())
    }

    fn quoted(&mut self) -> Result<String, RlError> {
        self.bump(); // opening quote
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return self.err("unterminated quoted value"),
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some(c @ ('"' | '\\')) => out.push(c),
                    _ => return self.err("invalid escape in quoted value"),
                },
                Some(c) => out.push(c),
            }
        }
    }

    fn value_list(&mut self) -> Result<Vec<String>, RlError> {
        self.skip_ws();
        if !self.eat("(") {
            return self.err("expected `(`");
        }
        let mut values = Vec::new();
        loop {
            self.skip_ws();
            values.push(self.value()?);
            self.skip_ws();
            if self.eat(")") {
                return Ok(values);
            }
            if !self.eat(",") {
                return self.err("expected `,` or `)`");
            }
        }
    }

    fn term(&mut self) -> Result<Term, RlError> {
        self.skip_ws();
        let key = self.key()?;
        self.skip_ws();
        let op = self.op()?;
        let values = match op {
            Op::Eq | Op::Ne => {
                self.skip_ws();
                vec![self.value()?]
            }
            Op::In | Op::NotIn => self.value_list()?,
        };
        Ok(Term { key, op, values })
    }
}

impl Selector {
    /// Parse a selector expression. Errors carry the byte offset of the problem.
    pub fn parse(src: &str) -> Result<Self, RlError> {
        let mut sc = Scanner { src, pos: 0 };
        let mut terms = Vec::new();
        loop {
            terms.push(sc.term()?);
            sc.skip_ws();
            if sc.pos == src.len() {
                break;
            }
            if !sc.eat(",") {
                return sc.err("expected `,` or end of selector");
            }
            sc.skip_ws();
            if sc.pos == src.len() {
                return sc.err("expected key after `,`");
            }
        }
        Ok(Self {
            source: src.to_string(),
            terms,
        })
    }

    /// The original expression text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Exact-match evaluation against a merged label view.
    pub fn matches(&self, view: &HashMap<String, String>) -> bool {
        self.terms.iter().all(|t| {
            let actual = view.get(&t.key);
            match t.op {
                Op::Eq => actual == t.values.first(),
                Op::Ne => actual != t.values.first(),
                Op::In => actual.is_some_and(|a| t.values.iter().any(|v| v == a)),
                Op::NotIn => !actual.is_some_and(|a| t.values.iter().any(|v| v == a)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
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
}
