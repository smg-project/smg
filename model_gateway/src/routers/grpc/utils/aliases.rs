//! The names an operator gives the `--tool-call-parser` and `--reasoning-parser` flags, and a
//! worker's `tool_parser` and `reasoning_parser` labels: a Symphony table's own name, or one of
//! the other names a table answers to ([`answering_to`]: the gateway's older parser names whose
//! format a table reads, `qwen` for `qwen3`, `qwen_xml` for `qwen3_tagged`, and a launcher's
//! name for a family whose table has no older parser), so a deployment's existing flags keep
//! naming a working table. Of the older names no table answers to ([`UNANSWERED`]), a tool-call
//! name is refused at startup and at a worker's registration with its reason, and a reasoning
//! name passes and selects nothing, both places saying so with its reason. The crate knows a
//! table by its own name only (`Table::named`): these names are the gateway's, and live with
//! its parser resolution.

use symphony::formats::Table;

/// The table that answers to `name`: the table named so ([`Table::named`]), else the one whose
/// format the older parser of that name reads, or that a launcher names so ([`ALIASES`]); `None`
/// for any other text.
pub fn answering_to(name: &str) -> Option<Table> {
    Table::named(name).or_else(|| {
        let name = name.trim().to_ascii_lowercase();
        let (_, table) = ALIASES.iter().find(|(alias, _)| *alias == name)?;
        Some(*table)
    })
}

/// The other names a table answers to ([`answering_to`]): the gateway's older tool and reasoning
/// parsers that read the table's format, so that a deployment that names one (`--tool-call-parser
/// qwen_xml`) reads with the table (`qwen3_tagged`); every older parser's name whose format a
/// table reads is an alias of that table, and an older name no table answers to is in
/// [`UNANSWERED`] with the reason. A row may also be the name a downstream launcher passes for a
/// model family whose table has no older parser (`muse` for `muse_spark`), so that the launcher's
/// flags read with the table, or an engine's exact name for a table (`llama4_pythonic` for
/// `llama4`, whose generic older name `pythonic` is in [`UNANSWERED`] while another family carries
/// it). `passthrough`, the older parser that left its channel as written, names the `plain`
/// table, which reads the whole output as content.
pub const ALIASES: [(&str, Table); 21] = [
    ("qwen", Table::Qwen3),
    ("qwen_xml", Table::Qwen3Tagged),
    ("qwen_coder", Table::Qwen3Tagged),
    ("nemotron", Table::Qwen3Tagged),
    ("glm47_moe", Table::Glm),
    ("glm45_moe", Table::Glm4),
    ("hy_v4", Table::Hy4),
    ("deepseek31", Table::DeepSeekV3_1),
    ("deepseek_v31", Table::DeepSeekV3_1),
    ("deepseek", Table::DeepSeekV3),
    // The R1 line's older reasoning name: V3's table reads its thought and the V3 calls the line
    // writes after it, until the line has a table of its own; startup says so ([`NOTED`]).
    ("deepseek_r1", Table::DeepSeekV3),
    ("deepseek32", Table::DeepSeekV3_2),
    ("deepseek_v41", Table::DeepSeekV4_1),
    ("llama4_pythonic", Table::Llama4),
    ("minimax", Table::MinimaxM2),
    ("muse", Table::MuseSpark),
    ("kimik2", Table::KimiK2),
    ("kimi_thinking", Table::KimiK2),
    ("kimi_k25", Table::KimiK2),
    // Llama 3.2's calls behind the python tag, `;` between them: the format Llama 3.1's models
    // write, which the json table reads. One shape differs: the older parser took the tag
    // anywhere in the text, prose before it as content; the table reads it at the output's
    // start only, as the engine's own reader of the family does, and an output with prose
    // before the tag is content whole (`docs/tables.md`, the JSON table).
    ("llama", Table::Json),
    // The output as written, content and nothing else: what the older `passthrough` parser left,
    // and what a model nothing else reads gets.
    ("passthrough", Table::Plain),
];

/// The older names a table answers to with a word at startup, each with what the operator learns:
/// the family's own table is pending, so the name reads with another family's table for now
/// (`regular::symphony::noted_name`). A row leaves when the family's table arrives and
/// [`ALIASES`] points the name at it.
pub const NOTED: [(&str, &str); 1] = [(
    "deepseek_r1",
    "DeepSeek R1's thought, which the R1 distills share, read with DeepSeek V3's table, which \
     reads the V3 calls the line writes after it, until the R1 line has a table of its own",
)];

/// The gateway's older parser names no table answers to ([`answering_to`] gives `None`), each
/// with the reason. A tool-call name here is refused at startup and at a worker's registration,
/// with the reason beside the tables an operator may name (`regular::symphony::check_names`), so
/// an operator who kept a flag from before the tables learns which table reads the family now. A
/// reasoning name here passes and selects nothing, since the thought it named is a family's the
/// tool-call name or the registry decides; startup and the registration say so with the reason
/// (`regular::symphony::unanswered_reasoning`). A table that takes a name here removes its row.
pub const UNANSWERED: [(&str, &str); 10] = [
    (
        "base",
        "a thought between `<think>` and `</think>` and nothing of a family's: the \
         registry's table reads a known model's thought, and an unknown model's the table \
         the flags name, else `plain`",
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
        "pythonic",
        "Llama 4's Python list of calls, which the llama4 table reads, and the name the \
         DeepSeek R1 line carries while its own table is pending: the alias arrives once \
         every family that names pythonic has a table; llama4_pythonic names the Llama 4 \
         table exactly",
    ),
    (
        "kimi",
        "a thought between `◁think▷` and `◁/think▷`, which no table reads: the Kimi K2 \
         line's table reads `<think>`",
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
        assert_eq!(answering_to("llama"), Some(Table::Json));
        assert_eq!(answering_to("pythonic"), None);
        // The older name of reading nothing: the plain table, by another name.
        assert_eq!(answering_to("passthrough"), Some(Table::Plain));
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
