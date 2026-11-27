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
/// with Symphony. The rule, so that a deployment's existing flags keep a working parser under
/// Symphony, the gateway's default parser: every legacy tool-call or reasoning parser's name
/// whose format a table reads is an alias of that table; a legacy name no table answers to is
/// in [`UNANSWERED`] with the reason (what it selects then is said there), `passthrough`
/// among them, which leaves its channel as written, which no table reads for. A row may also be
/// the name a downstream launcher passes for a model family whose table has no older parser
/// (`muse` for `muse_spark`), so that the launcher's flags read with the table, or an engine's
/// exact name for a table (`llama4_pythonic` for `llama4`, whose generic legacy name `pythonic`
/// is in [`UNANSWERED`] while another family carries it). One row names a family, not a format:
/// `minimax` reads with the table of the family's current line (`minimax_m3`), though the
/// legacy reasoning parser of that name reads the M2 line's thought; a deployment of that line
/// names its table by the table's own name, `minimax_m2`, where the registry does not list
/// its model: alone, the name gives way to the registry ([`gives_way`]).
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
    ("deepseek32", Table::DeepSeekV3_2),
    ("deepseek_v41", Table::DeepSeekV4_1),
    ("llama4_pythonic", Table::Llama4),
    ("minimax", Table::MinimaxM3),
    ("muse", Table::MuseSpark),
    // Command A's and Command R v01's calls, and Command A's thought. Where the table reads
    // otherwise than the legacy parsers of the two names read: `docs/tables.md`, the
    // Command table.
    ("cohere", Table::Command),
    ("cohere_cmd", Table::Command),
    ("kimik2", Table::KimiK2),
    ("kimi_thinking", Table::KimiK2),
    ("kimi_k25", Table::KimiK2),
    // Llama 3.2's calls behind the python tag, `;` between them: the format Llama 3.1's models
    // write, which the json table reads. One shape differs: the legacy parser took the tag
    // anywhere in the text, prose before it as content; the table reads it at the output's
    // start only, as the engine's own reader of the family does, and an output with prose
    // before the tag is content whole (`docs/tables.md`, the JSON table).
    ("llama", Table::Json),
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
pub const UNANSWERED: [(&str, &str); 9] = [
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
        "deepseek_r1",
        "DeepSeek R1's thought, which the R1 distills share; the tool-call name decides: \
         deepseek reads the R1 line with DeepSeek V3's table",
    ),
    (
        "pythonic",
        "Llama 4's Python list of calls, which the llama4 table reads, and the name the \
         DeepSeek R1 line carries while its own table is pending: the alias arrives once \
         every family that names pythonic has a table; llama4_pythonic names the Llama 4 \
         table exactly",
    ),
    (
        "step3",
        "Step 3's calls and thought, which the registry reads with the steptml table; the legacy \
         parser types a value by its look, the table by what the tool declares: no alias yet",
    ),
    ("sarashina", "Sarashina's calls: no table"),
];

/// The legacy reasoning parsers' names that give way where they stand alone ([`gives_way`]): a
/// table answers to each, and before the table came each selected nothing, its parser reading
/// the thought beside whatever read the calls. A source that names one and no tool-call parser
/// still selects nothing while another reader is left: the registry's table reads a model it
/// knows, and what the next source's names select reads the others. The name's table reads
/// where nothing else does, in the place of the legacy parsers. `minimax`, a family's name with
/// two lines, joined them so that the registry, which tells the lines apart, reads a model of
/// either line it lists; alone, it selected the M2 line's table before.
pub const GIVING_WAY: [&str; 3] = ["kimi", "cohere_cmd", "minimax"];

/// Whether `name` is a reasoning parser's that gives way where it stands alone
/// (`GIVING_WAY`).
pub fn gives_way(name: &str) -> bool {
    let name = name.trim();
    GIVING_WAY
        .iter()
        .any(|legacy| legacy.eq_ignore_ascii_case(name))
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/utils/aliases/tests.rs"]
mod tests;
