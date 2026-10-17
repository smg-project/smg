//! The gateway's own members of a response, under one `smg` object next to `usage`: absent
//! unless the request asked for them or the gateway's policy surfaces them, so every other
//! response stays wire-identical to what it was.

use serde::{Deserialize, Serialize};

/// The `smg` object of a response. The response types hold it boxed: it is absent on nearly
/// every response, and a streamed chunk is sized for what it carries on every frame.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SmgExtension {
    /// How the output was read, when the request asked with `X-SMG-Parse-Report` or the
    /// gateway's parse-failure policy surfaces it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parse: Option<ParseReport>,
}

impl SmgExtension {
    /// The object carrying `parse`.
    pub fn parse(report: ParseReport) -> Self {
        Self {
            parse: Some(report),
        }
    }

    /// The object carrying `parse`, boxed as the response types hold it.
    pub fn boxed_parse(report: ParseReport) -> Box<Self> {
        Box::new(Self::parse(report))
    }
}

/// What the parser that read the output says about it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ParseReport {
    /// The parser that read the output: `symphony` or `legacy`.
    pub parser: String,
    /// The table that read it, under `symphony`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table: Option<String>,
    /// `clean`, or `malformed` when the table gave up on a run of the output, under
    /// `symphony`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// The runs the table gave up on, in the output's order; their bytes are in the content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub malformed: Vec<ParseRun>,
    /// The runs the table dropped as the template's wrapping, when the request asked for
    /// `detail`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped: Vec<ParseRun>,
    /// Tool calls after the first that the gateway dropped under `parallel_tool_calls: false`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub suppressed_calls: u32,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes &T"
)]
fn is_zero(count: &u32) -> bool {
    *count == 0
}

/// One run of the output a table did not read as content, reasoning or a call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ParseRun {
    /// The choice the run belongs to.
    pub choice: u32,
    /// A malformed run's reason (`unterminated_region`, `invalid_arguments`, `unknown_tool`,
    /// `other`); a dropped run's kind (`control_token`, `wrapper`, `whitespace`).
    pub reason: String,
    /// The parser's own words for a reason the label does not spell out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    /// Where the run starts in the output, in bytes.
    pub at: u64,
    /// Its length in bytes.
    pub bytes: u64,
    /// Its first bytes, at most 120.
    pub excerpt: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_object_serializes_to_nothing_but_braces() {
        assert_eq!(
            serde_json::to_string(&SmgExtension::default()).expect("serializes"),
            "{}"
        );
    }

    #[test]
    fn a_report_leaves_out_what_it_does_not_have() {
        let report = ParseReport {
            parser: "legacy".into(),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_string(&SmgExtension::parse(report)).expect("serializes"),
            r#"{"parse":{"parser":"legacy"}}"#
        );
        let report = ParseReport {
            parser: "symphony".into(),
            table: Some("qwen3".into()),
            outcome: Some("malformed".into()),
            malformed: vec![ParseRun {
                choice: 0,
                reason: "invalid_arguments".into(),
                why: None,
                at: 6,
                bytes: 12,
                excerpt: "\nnot a call\n".into(),
            }],
            dropped: Vec::new(),
            suppressed_calls: 1,
        };
        let json = serde_json::to_value(SmgExtension::parse(report)).expect("serializes");
        assert_eq!(json["parse"]["malformed"][0]["at"], 6);
        assert_eq!(json["parse"]["suppressed_calls"], 1);
        assert!(json["parse"].get("dropped").is_none());
        let back: SmgExtension = serde_json::from_value(json).expect("deserializes");
        assert_eq!(back.parse.expect("parse").malformed.len(), 1);
    }
}
