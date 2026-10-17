//! Which table reads which checkpoint.
//!
//! [`for_model`] takes a model id, the Hugging Face id or the name a server gives the model, and
//! gives the [`Format`] that reads its output: the table of the checkpoint group the model belongs
//! to, with the group's own turn opener where its template writes one the table does not expect.
//! `None` means the registry does not know the model, and the caller reads its output another
//! way; the gateway keeps its older parsers for such a model.
//!
//! The groups are bellwether's: the checkpoints that share a tokenizer and a template, recorded
//! once by the group's primary. Every group whose table is on main is here, its primary and its
//! other checkpoints under the primary's organisation, spelled in `groups.json` beside this
//! module (one group per line; the crate's build script writes the file as the `GROUPS`
//! constant); `tests/readme.rs` writes the README's models table from the same groups and checks
//! that the two agree.
//!
//! A lookup matches the repository name, the text after the last path separator (either slash),
//! exactly and without regard to case, under any organisation or path: `Qwen/Qwen3-8B`,
//! `qwen3-8b` and a local path ending in `Qwen3-8B` all read with the same table. Unlike the
//! gateway's parser registries, which match a pattern anywhere in the name, a quantized or
//! renamed checkpoint (`Qwen3-8B-FP8`) is not matched and gives `None`, so the gateway keeps its
//! older parsers for it until its group lists the name under `also`.

use std::sync::{Arc, LazyLock, OnceLock};

use super::{
    deepseek_v4, deepseek_v4_1, dots3, function_gemma, gemma4, glm, glm4, hy4, inkling, iquest,
    json, k2_horizon, kimi_k3, lfm2_5, ling, llama4, llama_nemotron, minicpm5, minimax_m2,
    minimax_m3, mistral_v13, muse_spark, olmo3, plain, qwen2_5, qwen3, seed_oss, xlam,
};
use crate::{
    format::{CallSyntax, Format},
    tagged::{Placement, Spelling},
};

/// The tables, each spelled once: the variant with its doc, the name an operator spells it by
/// and the function that builds it. [`Table`] and [`ROWS`] are written from this one list, so a
/// table cannot be added without its row, wherever in the list it goes; the rows stand in the
/// variants' order, at their discriminants, and their count is the list's.
macro_rules! tables {
    ($( $(#[$doc:meta])* $variant:ident => $name:literal, $build:expr; )*) => {
        /// A table under [`formats`](super), by the function that builds it and, where the function
        /// takes one, its call syntax. [`Table::name`] is how an operator names one: the gateway's
        /// `--tool-call-parser` and `--reasoning-parser` flags, and a worker's labels, take a table's
        /// name in place of a legacy parser's.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Table {
            $( $(#[$doc])* $variant, )*
        }

        /// Every table's row, in the enum's order: the one place a table is spelled after its
        /// variant. [`Table::ALL`], [`Table::name`], [`Table::format`] and the lookups by name
        /// read it; a table reads its own by `self as usize`.
        const ROWS: [Row; tables!(@count $($variant)*)] = [
            $( row(Table::$variant, $name, $build), )*
        ];
    };
    (@count $($variant:ident)*) => { <[()]>::len(&[$( tables!(@unit $variant) ),*]) };
    (@unit $variant:ident) => { () };
}

tables! {
    /// [`qwen3()`] with JSON calls.
    Qwen3 => "qwen3", || qwen3(CallSyntax::Json);
    /// [`qwen3()`] with tagged calls, each value on its own line: Qwen 3.5 and later, Qwen3-Coder.
    Qwen3Tagged => "qwen3_tagged", || {
        qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine))
    };
    /// [`qwen3()`] with tagged calls and the value between the tags directly: MiMo.
    Qwen3TaggedDirect => "qwen3_tagged_direct", || {
        qwen3(CallSyntax::Tagged(Spelling::Json, Placement::Direct))
    };
    /// [`qwen2_5()`].
    Qwen2_5 => "qwen2.5", qwen2_5;
    /// [`deepseek_v4()`].
    DeepSeekV4 => "deepseek_v4", deepseek_v4;
    /// [`deepseek_v4_1()`].
    DeepSeekV4_1 => "deepseek_v4.1", deepseek_v4_1;
    /// [`seed_oss()`].
    SeedOss => "seed_oss", seed_oss;
    /// [`hy4()`].
    Hy4 => "hy4", hy4;
    /// [`ling()`].
    Ling => "ling", ling;
    /// [`iquest()`].
    IQuest => "iquest", iquest;
    /// [`olmo3()`].
    Olmo3 => "olmo3", olmo3;
    /// [`lfm2_5()`].
    Lfm2_5 => "lfm2.5", lfm2_5;
    /// [`xlam()`].
    Xlam => "xlam", xlam;
    /// [`json()`].
    Json => "json", json;
    /// [`minimax_m2()`].
    MinimaxM2 => "minimax_m2", minimax_m2;
    /// [`minimax_m3()`].
    MinimaxM3 => "minimax_m3", minimax_m3;
    /// [`mistral_v13()`].
    MistralV13 => "mistral_v13", mistral_v13;
    /// [`llama_nemotron()`].
    LlamaNemotron => "llama_nemotron", llama_nemotron;
    /// [`kimi_k3()`].
    KimiK3 => "kimi_k3", kimi_k3;
    /// [`inkling()`].
    Inkling => "inkling", inkling;
    /// [`minicpm5()`].
    MiniCpm5 => "minicpm5", minicpm5;
    /// [`dots3()`].
    Dots3 => "dots3", dots3;
    /// [`gemma4()`].
    Gemma4 => "gemma4", gemma4;
    /// [`function_gemma()`].
    FunctionGemma => "function_gemma", function_gemma;
    /// [`glm()`].
    Glm => "glm", glm;
    /// [`glm4()`].
    Glm4 => "glm4", glm4;
    /// [`k2_horizon()`].
    K2Horizon => "k2_horizon", k2_horizon;
    /// [`muse_spark()`].
    MuseSpark => "muse_spark", muse_spark;
    /// [`plain()`].
    Plain => "plain", plain;
    /// [`llama4()`].
    Llama4 => "llama4", llama4;
}

/// A table's row in the registry: the table, its name and the function that builds it. The rows
/// stand at the tables' discriminants, so that a table reads its own by `self as usize`.
struct Row {
    table: Table,
    name: &'static str,
    build: fn() -> Format,
}

/// A row ([`ROWS`]).
const fn row(table: Table, name: &'static str, build: fn() -> Format) -> Row {
    Row { table, name, build }
}

impl Table {
    /// Every table, in the rows' order, which is the enum's.
    pub const ALL: [Table; ROWS.len()] = {
        let mut all = [Table::Plain; ROWS.len()];
        let mut i = 0;
        while i < all.len() {
            all[i] = ROWS[i].table;
            i += 1;
        }
        all
    };

    /// The table's name, as an operator spells it: its format's name, with the call syntax after
    /// it where the function takes one (`qwen3`, `qwen3_tagged`, `qwen3_tagged_direct`).
    pub const fn name(self) -> &'static str {
        ROWS[self as usize].name
    }

    /// The table named `name`, a table's own name ([`Self::name`]) in any case; `None` for any
    /// other text, a legacy parser's name among them: [`Self::answering_to`] reads those.
    pub fn named(name: &str) -> Option<Self> {
        Self::exactly_named(&name.trim().to_ascii_lowercase())
    }

    /// The table named `name` exactly, as [`Self::name`] spells it; `None` for any other text. A
    /// constant function, so that a group can name its table where the groups are evaluated.
    const fn exactly_named(name: &str) -> Option<Self> {
        let mut i = 0;
        while i < ROWS.len() {
            if same(ROWS[i].name, name) {
                return Some(ROWS[i].table);
            }
            i += 1;
        }
        None
    }

    /// The table that answers to `name`: the table named so ([`Self::named`]), else the one whose
    /// format the legacy parser of that name reads ([`Self::ALIASES`]); `None` for any other text.
    pub fn answering_to(name: &str) -> Option<Self> {
        Self::named(name).or_else(|| {
            let name = name.trim().to_ascii_lowercase();
            let (_, table) = Self::ALIASES.iter().find(|(alias, _)| *alias == name)?;
            Some(*table)
        })
    }

    /// The other names a table answers to ([`Self::answering_to`]): the gateway's older tool and
    /// reasoning parsers that read the table's format, so that a deployment that names one
    /// (`--tool-call-parser qwen_xml`) reads with the table (`qwen3_tagged`) when the gateway
    /// reads with Symphony. A parser that reads a format no table reads has no row and keeps its
    /// own parser; so does `passthrough`, which leaves its channel as written, which no table
    /// reads for. A row may also be the name a downstream launcher passes for a model family
    /// whose table has no older parser (`muse` for `muse_spark`), so that the launcher's flags
    /// read with the table, or an engine's exact name for a table (`llama4_pythonic` for
    /// `llama4`), beside the generic legacy name.
    pub const ALIASES: [(&str, Table); 12] = [
        ("qwen", Table::Qwen3),
        ("qwen_xml", Table::Qwen3Tagged),
        ("qwen_coder", Table::Qwen3Tagged),
        ("nemotron", Table::Qwen3Tagged),
        ("glm47_moe", Table::Glm),
        ("glm45_moe", Table::Glm4),
        ("hy_v4", Table::Hy4),
        ("deepseek_v41", Table::DeepSeekV4_1),
        ("pythonic", Table::Llama4),
        ("llama4_pythonic", Table::Llama4),
        ("minimax", Table::MinimaxM2),
        ("muse", Table::MuseSpark),
    ];

    /// The table, built once and shared: the engines that run it, and whoever asks the table
    /// for its tags, read the one copy. [`Self::format`] builds a copy of one's own to change.
    /// The copies stand at the tables' positions in [`Self::ALL`], which is the enum's order;
    /// every table a name or a group can select is in `ALL` (the tests say so).
    pub fn shared(self) -> Arc<Format> {
        static SHARED: [OnceLock<Arc<Format>>; Table::ALL.len()] =
            [const { OnceLock::new() }; Table::ALL.len()];
        Arc::clone(SHARED[self as usize].get_or_init(|| Arc::new(self.format())))
    }

    /// The table, built.
    pub fn format(self) -> Format {
        (ROWS[self as usize].build)()
    }
}

/// A checkpoint group of bellwether's: the checkpoints that share a tokenizer and a template.
struct Group {
    /// The primary's Hugging Face id, the organisation and the repository name.
    model: &'static str,
    /// The group's other checkpoints: by repository name under the primary's organisation, by
    /// full id under another.
    also: &'static [&'static str],
    /// The table that reads the group's output.
    table: Table,
    /// What the group's own template opens the model's turn with, when it is not the table's
    /// own: the prompt's replay starts at its last occurrence.
    opener: Option<&'static str>,
}

impl Group {
    /// The group's checkpoints, by repository name: the primary first.
    fn repositories(&self) -> impl Iterator<Item = &'static str> {
        std::iter::once(repository(self.model)).chain(self.also.iter().map(|also| repository(also)))
    }

    /// The group's table, with the group's own turn opener where its template writes one.
    fn format(&self) -> Format {
        let format = self.table.format();
        match self.opener {
            Some(opener) => format.opens_turn(opener),
            None => format,
        }
    }
}

/// The repository name in `model`: the text after its last path separator, either slash, a
/// trailing one ignored.
fn repository(model: &str) -> &str {
    let model = model.trim_end_matches(['/', '\\']);
    model.rsplit(['/', '\\']).next().unwrap_or(model)
}

/// Whether `a` and `b` are the same text: `==` for a constant context.
const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The table named `name` ([`Table::name`]) where the groups are evaluated: a name the groups
/// file spells that no table has fails the build, at the group that spells it.
#[expect(
    clippy::panic,
    reason = "a group that names a table the registry does not have is a programming error, caught when the crate builds"
)]
const fn table_named(name: &str) -> Table {
    match Table::exactly_named(name) {
        Some(table) => table,
        None => panic!("groups.json names a table the registry does not have"),
    }
}

// Every group whose table is on main, in bellwether's slug order, as `const GROUPS: &[Group]`:
// `groups.json` beside this module is the one place a group is spelled (the primary's id, its
// other checkpoints, its table by name, its own turn opener), one group per line, and `build.rs`
// writes it as this constant, so the compiled tables are the literals they were.
include!(concat!(env!("OUT_DIR"), "/groups.rs"));

/// Every group's checkpoints by repository name, each with its group's index, in the registry's
/// order: worked out once, since a request looks its model up more than once.
static REPOSITORIES: LazyLock<Vec<(&'static str, usize)>> = LazyLock::new(|| {
    GROUPS
        .iter()
        .enumerate()
        .flat_map(|(index, group)| group.repositories().map(move |own| (own, index)))
        .collect()
});

/// Every group's table ([`Group::format`]), built once and shared, by the group's index.
static GROUP_FORMATS: [OnceLock<Arc<Format>>; GROUPS.len()] =
    [const { OnceLock::new() }; GROUPS.len()];

/// The group that holds `model`, by its repository name in any case: the first in the
/// registry's order, with its index.
fn group_for(model: &str) -> Option<(usize, &'static Group)> {
    let repository = repository(model);
    if repository.is_empty() {
        return None;
    }
    REPOSITORIES
        .iter()
        .find(|(own, _)| own.eq_ignore_ascii_case(repository))
        .map(|&(_, index)| (index, &GROUPS[index]))
}

/// The table that reads `model`'s output, with the model's own turn opener where its template has
/// one, as a copy of the caller's own; `None` for a model the registry does not know.
pub fn for_model(model: &str) -> Option<Format> {
    for_model_shared(model).map(|format| (*format).clone())
}

/// [`for_model`]'s table, built once and shared, as [`Table::shared`] shares a table: the engines
/// that read a model's output hold the one copy.
pub fn for_model_shared(model: &str) -> Option<Arc<Format>> {
    let (index, group) = group_for(model)?;
    Some(Arc::clone(
        GROUP_FORMATS[index].get_or_init(|| Arc::new(group.format())),
    ))
}

/// The table [`for_model`] builds for `model`, by name; `None` for a model the registry does not
/// know.
pub fn table_for(model: &str) -> Option<Table> {
    group_for(model).map(|(_, group)| group.table)
}

/// Every group the registry knows: the primary's id, its other checkpoints by repository name,
/// and the table, in the registry's order. The README test checks its groups against these.
pub fn groups() -> impl Iterator<Item = (&'static str, &'static [&'static str], Table)> {
    GROUPS
        .iter()
        .map(|group| (group.model, group.also, group.table))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Emits;

    /// `groups.json` as the registry spells it: one group per line, in the registry's order,
    /// every field written, so that a diff is a group and a tool records a group as one line.
    fn groups_json() -> String {
        fn text(text: &str) -> String {
            serde_json::to_string(text).expect("a JSON string")
        }
        let lines: Vec<String> = GROUPS
            .iter()
            .map(|group| {
                let also: Vec<String> = group.also.iter().map(|also| text(also)).collect();
                format!(
                    r#"  {{"model": {}, "also": [{}], "table": {}, "opener": {}}}"#,
                    text(group.model),
                    also.join(", "),
                    text(group.table.name()),
                    group.opener.map_or_else(|| "null".to_string(), text),
                )
            })
            .collect();
        format!("[\n{}\n]\n", lines.join(",\n"))
    }

    #[test]
    fn the_groups_file_is_written_one_group_per_line_in_the_registrys_form() {
        let expected = groups_json();
        assert!(
            include_str!("groups.json") == expected,
            "groups.json is not written in the registry's form; replace it with:\n{expected}"
        );
    }

    #[test]
    fn a_table_is_named_exactly_where_the_groups_are_evaluated() {
        for table in Table::ALL {
            assert_eq!(Table::exactly_named(table.name()), Some(table));
            assert_eq!(table_named(table.name()), table);
        }
        assert_eq!(
            Table::exactly_named("Qwen3"),
            None,
            "the exact spelling only"
        );
        assert_eq!(Table::exactly_named(" qwen3"), None);
        assert_eq!(
            Table::exactly_named("qwen"),
            None,
            "a legacy parser's name is no table"
        );
        assert_eq!(Table::exactly_named(""), None);
    }

    #[test]
    fn a_primary_resolves_by_its_id_in_any_case_with_or_without_the_organisation() {
        for model in [
            "Qwen/Qwen3-8B",
            "qwen/qwen3-8b",
            "QWEN3-8B",
            "Qwen3-8B",
            "/models/Qwen/Qwen3-8B/",
            "other-org/Qwen3-8B",
            r"C:\models\Qwen3-8B",
            r"C:\models\Qwen3-8B\",
        ] {
            assert_eq!(table_for(model), Some(Table::Qwen3), "{model}");
            let format = for_model(model).unwrap_or_else(|| panic!("{model}"));
            assert_eq!(format.name(), "qwen3", "{model}");
            assert_eq!(
                format.turn_opener(),
                Some("<|im_start|>assistant"),
                "{model}"
            );
        }
    }

    #[test]
    fn another_checkpoint_of_a_group_resolves_to_the_groups_table() {
        assert_eq!(table_for("Qwen/Qwen3-0.6B"), Some(Table::Qwen3));
        assert_eq!(table_for("qwen3.5-397b-a17b"), Some(Table::Qwen3Tagged));
        assert_eq!(
            table_for("Qwen/Qwen2.5-VL-72B-Instruct"),
            Some(Table::Qwen2_5)
        );
        // The v1 Nemotron checkpoints, whose own template renders no call, read with the table
        // their vendor's tool parser reads for; the CPT checkpoint keeps Llama 3.1's syntax.
        assert_eq!(
            table_for("nvidia/Llama-3_3-Nemotron-Super-49B-v1"),
            Some(Table::LlamaNemotron)
        );
        assert_eq!(
            table_for("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"),
            Some(Table::LlamaNemotron)
        );
        assert_eq!(
            table_for("nvidia/Llama-3_1-Nemotron-Ultra-253B-CPT-v1"),
            Some(Table::Json)
        );
        assert_eq!(
            table_for("thinkingmachines/Inkling-NVFP4"),
            Some(Table::Inkling)
        );
        for mimo in [
            "XiaomiMiMo/MiMo-V2.6-Flash-RL",
            "XiaomiMiMo/MiMo-V2.6-Pro-RL",
            "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
            "XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
            "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
        ] {
            assert_eq!(table_for(mimo), Some(Table::Qwen3TaggedDirect), "{mimo}");
        }
    }

    #[test]
    fn a_model_the_registry_does_not_know_resolves_to_none() {
        for model in [
            "Qwen/QwQ-32B",
            "Qwen/Qwen3-8B-FP8",
            "Qwen",
            "Qwen/",
            "",
            "/",
        ] {
            assert_eq!(table_for(model), None, "{model}");
            assert!(for_model(model).is_none(), "{model}");
        }
    }

    #[test]
    fn a_group_with_its_own_turn_opener_replays_from_it() {
        let exaone = for_model("LGAI-EXAONE/K-EXAONE-236B-A23B").expect("K-EXAONE");
        assert_eq!(exaone.name(), "qwen3");
        assert_eq!(exaone.turn_opener(), Some("<|assistant|>"));
        let granite = for_model("granite-4.1-3b").expect("Granite 4.1");
        assert_eq!(granite.name(), "qwen2.5");
        assert_eq!(
            granite.turn_opener(),
            Some("<|start_of_role|>assistant<|end_of_role|>")
        );
    }

    #[test]
    fn every_alias_spells_its_table_and_is_no_tables_own_name() {
        let mut seen = std::collections::HashSet::new();
        for (alias, table) in Table::ALIASES {
            assert!(seen.insert(alias), "{alias} is listed twice");
            assert!(
                Table::ALL.iter().all(|own| own.name() != alias),
                "{alias} is a table's own name"
            );
            assert_eq!(Table::named(alias), None, "{alias} is a table's own name");
            // Every table a name can select stands in `ALL`: `Table::shared` keeps its tables
            // at their positions there.
            assert!(Table::ALL.contains(&table), "{alias}'s table is not in ALL");
            assert_eq!(Table::answering_to(alias), Some(table), "{alias}");
            assert_eq!(
                Table::answering_to(&alias.to_ascii_uppercase()),
                Some(table),
                "{alias}"
            );
        }
        assert_eq!(Table::answering_to("llama"), None);
        assert_eq!(Table::answering_to("passthrough"), None);
        // A launcher's name for a family whose table has no older parser answers for the table.
        assert_eq!(Table::answering_to("muse"), Some(Table::MuseSpark));
        assert_eq!(Table::named("muse"), None);
    }

    #[test]
    fn every_table_has_a_name_of_its_own_that_spells_it_back() {
        let mut seen: Vec<&str> = Vec::new();
        for table in Table::ALL {
            let name = table.name();
            assert!(!seen.contains(&name), "{name} names two tables");
            seen.push(name);
            assert_eq!(Table::named(name), Some(table), "{name}");
            assert_eq!(
                Table::named(&format!(" {} ", name.to_uppercase())),
                Some(table)
            );
            // A format built once per table: the name is the format's where the function takes
            // no call syntax, and the format's with the syntax after it where it does.
            let format = table.format();
            if matches!(table, Table::Qwen3Tagged | Table::Qwen3TaggedDirect) {
                assert!(name.starts_with(&format!("{}_", format.name())), "{name}");
            } else {
                assert_eq!(name, format.name());
            }
        }
        assert_eq!(seen.len(), Table::ALL.len());
        assert_eq!(
            Table::named("qwen"),
            None,
            "a legacy parser's name is no table"
        );
        assert_eq!(Table::named(""), None);
        for group in GROUPS {
            assert!(Table::ALL.contains(&group.table), "{}", group.model);
        }
    }

    #[test]
    fn a_shared_table_is_one_copy_for_everyone_and_equal_to_a_fresh_build() {
        for table in Table::ALL {
            let shared = table.shared();
            assert!(Arc::ptr_eq(&shared, &table.shared()), "{}", table.name());
            assert_eq!(*shared, table.format(), "{}", table.name());
        }
        for (model, _, _) in groups() {
            let shared = for_model_shared(model).unwrap_or_else(|| panic!("{model}"));
            let again = for_model_shared(model).unwrap_or_else(|| panic!("{model}"));
            assert!(Arc::ptr_eq(&shared, &again), "{model}");
            assert_eq!(Some(&*shared), for_model(model).as_ref(), "{model}");
        }
    }

    #[test]
    fn every_table_stands_at_its_own_index_in_all() {
        for (index, table) in Table::ALL.into_iter().enumerate() {
            assert_eq!(table as usize, index, "{}", table.name());
        }
    }

    #[test]
    fn every_table_builds_and_takes_its_groups_opener() {
        for group in GROUPS {
            let format = for_model(group.model).unwrap_or_else(|| panic!("{}", group.model));
            assert_eq!(format.validate(), Ok(()), "{}", group.model);
            let own = group.table.format();
            let expected = group.opener.or_else(|| own.turn_opener());
            assert_eq!(format.turn_opener(), expected, "{}", group.model);
        }
    }

    #[test]
    fn every_primary_has_an_organisation_and_every_repository_name_is_unique() {
        let mut seen: Vec<String> = Vec::new();
        for group in GROUPS {
            assert_eq!(
                group.model.matches('/').count(),
                1,
                "{}: not an organisation and a repository",
                group.model
            );
            for repository in group.repositories() {
                let lower = repository.to_ascii_lowercase();
                assert!(!seen.contains(&lower), "{repository} is listed twice");
                seen.push(lower);
            }
        }
    }

    #[test]
    fn every_tables_variants_without_reasoning_or_calls_still_validate() {
        for table in Table::ALL {
            let format = table.format();
            let silent = format.clone().without_reasoning();
            silent
                .validate()
                .unwrap_or_else(|e| panic!("{}: without_reasoning: {e}", table.name()));
            assert!(
                (0..silent.states()).all(|state| silent.emits(state) != Emits::Reasoning),
                "{}: a thought state remains",
                table.name()
            );
            let unread = format.without_calls();
            unread
                .validate()
                .unwrap_or_else(|e| panic!("{}: without_calls: {e}", table.name()));
            assert_eq!(unread.call_syntax(), None, "{}", table.name());
            assert!(
                (0..unread.states()).all(|state| unread.emits(state) != Emits::Arguments),
                "{}: an arguments state remains",
                table.name()
            );
            assert!(
                (1..unread.states()).all(|state| unread.emits(state) != Emits::Wrapper),
                "{}: a calls wrapper remains",
                table.name()
            );
        }
    }
}
