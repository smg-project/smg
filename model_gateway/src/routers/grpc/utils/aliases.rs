//! The gateway's older parser names and the Symphony tables that answer to them.
//!
//! The `--tool-call-parser` and `--reasoning-parser` flags, and a worker's labels, name the older
//! parsers. Under `--parser symphony` a name whose format a table reads selects that table
//! ([`answering_to`]: `qwen` for `qwen3`, `qwen_xml` for `qwen3_tagged`), so a deployment's
//! existing flags keep a working parser when the gateway reads with Symphony; the rule is that
//! every legacy name is a table's own, one of its [`ALIASES`], or in [`UNANSWERED`] with the
//! reason, never both and never neither, which the test over both legacy registries holds
//! (`regular::symphony`). The crate knows a table by its own name only (`Table::named`): these
//! names are the gateway's, and live with its parser resolution.

use symphony::formats::Table;

/// The table that answers to `name`: the table named so ([`Table::named`]), else the one whose
/// format the legacy parser of that name reads ([`ALIASES`]); `None` for any other text.
pub fn answering_to(name: &str) -> Option<Table> {
    Table::named(name).or_else(|| {
        let name = name.trim().to_ascii_lowercase();
        let (_, table) = ALIASES.iter().find(|(alias, _)| *alias == name)?;
        Some(*table)
    })
}

/// The other names a table answers to ([`answering_to`]): the gateway's older tool and
/// reasoning parsers that read the table's format, so that a deployment that names one
/// (`--tool-call-parser qwen_xml`) reads with the table (`qwen3_tagged`) when the gateway reads
/// with Symphony. The rule, so that a deployment's existing flags keep a working parser when
/// the gateway's default parser changes: every legacy tool-call or reasoning parser's name
/// whose format a table reads is an alias of that table; a legacy name no table answers to is
/// in [`UNANSWERED`] with the reason (what it selects then is said there), `passthrough`
/// among them, which leaves its channel as written, which no table reads for. A row may also be
/// the name a downstream launcher passes for a model family whose table has no older parser
/// (`muse` for `muse_spark`), so that the launcher's flags read with the table.
pub const ALIASES: [(&str, Table); 14] = [
    ("deepseek32", Table::DeepSeekV3_2),
    ("qwen", Table::Qwen3),
    ("qwen_xml", Table::Qwen3Tagged),
    ("qwen_coder", Table::Qwen3Tagged),
    ("nemotron", Table::Qwen3Tagged),
    ("glm47_moe", Table::Glm),
    ("glm45_moe", Table::Glm4),
    ("hy_v4", Table::Hy4),
    ("deepseek_v41", Table::DeepSeekV4_1),
    ("minimax", Table::MinimaxM2),
    ("muse", Table::MuseSpark),
    ("kimik2", Table::KimiK2),
    ("kimi_thinking", Table::KimiK2),
    ("kimi_k25", Table::KimiK2),
];

/// The legacy parsers' names no table answers to ([`answering_to`] gives `None`), each
/// with the reason, so that every name of the gateway's two parser registries is accounted
/// for: a legacy name is here or is one a table answers to, never both and never neither,
/// which the gateway's test over its registries holds; so a new legacy name cannot go
/// unmapped, and a table that takes a name here removes its row. Under Symphony a tool-call
/// name here keeps the legacy parsers, and a reasoning name here alone selects nothing, so
/// the next source decides: after a worker's labels the registry's table for a model it
/// knows, else the gateway's flags, else the legacy parsers (`passthrough` excepted, which
/// keeps the legacy parsers either way).
pub const UNANSWERED: [(&str, &str); 17] = [
    (
        "passthrough",
        "leaves its channel as written, which no table reads for: the way back to the \
         legacy parsers for one model",
    ),
    (
        "base",
        "a thought between `<think>` and `</think>` and nothing of a family's: the \
         registry's table reads a known model's thought, and an unknown model's the table \
         the flags name, else the legacy parser",
    ),
    (
        "glm45",
        "a thought two tables' families share, glm4 (GLM 4.5 and 4.6) and glm (GLM 4.7 and \
         later): the tool-call name, glm45_moe or glm47_moe, or the registry decides",
    ),
    (
        "nano_v3",
        "a thought two tables' families share, qwen3_tagged (Nemotron 3) and llama_nemotron \
         (Llama Nemotron): the tool-call name, nemotron or json, or the registry decides",
    ),
    (
        "qwen3_thinking",
        "a thought two tables' families share, qwen3 (the Qwen3-2507 thinking line) and \
         qwen3_tagged (Qwen 3.5 and later): the tool-call name, qwen or qwen_xml, or the \
         registry decides",
    ),
    (
        "deepseek",
        "DeepSeek V3's call spelling, which no table reads yet; the table that does takes \
         the name",
    ),
    (
        "deepseek31",
        "DeepSeek V3.1's call spelling, which no table reads yet; the table that does takes \
         the name",
    ),
    (
        "deepseek_v31",
        "DeepSeek V3.1's thought, whose table does not exist yet; the table takes the name",
    ),
    (
        "deepseek_r1",
        "DeepSeek R1's thought; the R1 line reads with DeepSeek V3's table, which does not \
         exist yet and takes the name",
    ),
    (
        "llama",
        "Llama 3.2's calls after its python tag, which no table reads yet; the table that \
         does takes the name",
    ),
    (
        "pythonic",
        "Llama 4's Python list of calls, which no table reads yet; the table that does takes \
         the name",
    ),
    (
        "kimi",
        "a thought between `◁think▷` and `◁/think▷`, which no table reads: the Kimi K2 \
         line's table reads `<think>`",
    ),
    (
        "mistral",
        "the list of calls after `[TOOL_CALLS]` of Mistral 7B v0.3 and Small 3, another \
         spelling than mistral_v13's, which no table reads yet; the table that does takes \
         the name",
    ),
    ("step3", "Step 3's calls and thought: no table"),
    ("sarashina", "Sarashina's calls: no table"),
    ("cohere", "Command R and Command A's calls: no table"),
    ("cohere_cmd", "Command R and Command A's thought: no table"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_alias_spells_its_table_and_is_no_tables_own_name() {
        let mut seen = std::collections::HashSet::new();
        for (alias, table) in ALIASES {
            assert!(seen.insert(alias), "{alias} is listed twice");
            assert!(
                Table::ALL.iter().all(|own| own.name() != alias),
                "{alias} is a table's own name"
            );
            assert_eq!(Table::named(alias), None, "{alias} is a table's own name");
            // Every table a name can select stands in `ALL`: `Table::shared` keeps its tables
            // at their positions there.
            assert!(Table::ALL.contains(&table), "{alias}'s table is not in ALL");
            assert_eq!(answering_to(alias), Some(table), "{alias}");
            assert_eq!(
                answering_to(&alias.to_ascii_uppercase()),
                Some(table),
                "{alias}"
            );
        }
        assert_eq!(answering_to("llama"), None);
        assert_eq!(answering_to("passthrough"), None);
        // A launcher's name for a family whose table has no older parser answers for the table.
        assert_eq!(answering_to("muse"), Some(Table::MuseSpark));
        assert_eq!(Table::named("muse"), None);
    }

    #[test]
    fn every_unanswered_name_answers_to_no_table_and_says_why() {
        let mut seen = std::collections::HashSet::new();
        for (name, reason) in UNANSWERED {
            assert!(seen.insert(name), "{name} is listed twice");
            assert_eq!(
                name,
                name.trim().to_ascii_lowercase(),
                "{name} is not spelled as the legacy registries spell their names"
            );
            assert_eq!(Table::named(name), None, "{name} is a table's own name");
            assert_eq!(
                answering_to(name),
                None,
                "{name} answers to a table: remove its row"
            );
            assert!(!reason.trim().is_empty(), "{name} has no reason");
        }
    }
}
