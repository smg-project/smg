//! Which table reads which checkpoint.
//!
//! [`for_model`] takes a model id, the Hugging Face id or the name a server gives the model, and
//! gives the [`Format`] that reads its output: the table of the checkpoint group the model belongs
//! to, with the group's own turn opener where its template writes one the table does not expect,
//! and the text its template writes after the thought's close where that is not the table's.
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

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{Arc, LazyLock, OnceLock},
};

use super::{
    deepseek_v3, deepseek_v3_1, deepseek_v3_2, deepseek_v4, deepseek_v4_1, dots3, function_gemma,
    gemma4, glm, glm4, hy4, inkling, iquest, json, k2_horizon, kimi_k2, kimi_k3, lfm2_5, ling,
    llama4, llama_nemotron, minicpm5, minimax_m2, minimax_m3, mistral, mistral_v13, muse_spark,
    olmo3, plain, qwen2_5, qwen3, seed_oss, xlam,
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
    /// [`deepseek_v3()`].
    DeepSeekV3 => "deepseek_v3", deepseek_v3;
    /// [`deepseek_v3_1()`].
    DeepSeekV3_1 => "deepseek_v3.1", deepseek_v3_1;
    /// [`deepseek_v3_2()`].
    DeepSeekV3_2 => "deepseek_v3.2", deepseek_v3_2;
    /// [`llama4()`].
    Llama4 => "llama4", llama4;
    /// [`mistral()`].
    Mistral => "mistral", mistral;
    /// [`kimi_k2()`].
    KimiK2 => "kimi_k2", kimi_k2;
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
    /// other text. The gateway's older parser names are the gateway's to map (`qwen` for `qwen3`):
    /// the crate knows a table by its own name.
    pub fn named(name: &str) -> Option<Self> {
        let name = name.trim();
        ROWS.iter()
            .find(|row| row.name.eq_ignore_ascii_case(name))
            .map(|row| row.table)
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
    /// What the group's own template writes after the thought's close before the first call,
    /// when it is not the table's own (the NVIDIA Nemotron 3 groups and Step 3.5 Flash write one
    /// newline where the Qwen 3.5 template writes two): the group's tags write it in the table's place
    /// ([`Format::with_thought_after`]).
    after: Option<&'static str>,
}

impl Group {
    /// The group's checkpoints, by repository name: the primary first.
    fn repositories(&self) -> impl Iterator<Item = &'static str> {
        std::iter::once(repository(self.model)).chain(self.also.iter().map(|also| repository(also)))
    }

    /// The group's table, with the group's own turn opener where its template writes one, and
    /// its own text after the thought's close where its template writes other than the table's.
    fn format(&self) -> Format {
        let format = self.table.format();
        let format = match self.opener {
            Some(opener) => format.opens_turn(opener),
            None => format,
        };
        match self.after {
            Some(after) => format.with_thought_after(after),
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
// other checkpoints, its table by name, its own turn opener, its own text after the thought's
// close where the template writes other than the table's), one group per line, and `build.rs`
// writes it as this constant, so the compiled tables are the literals they were.
include!(concat!(env!("OUT_DIR"), "/groups.rs"));

/// Every group's checkpoints by repository name, each with its group's index, keyed without
/// regard to ASCII case ([`Caseless`]): worked out once, since a request looks its model up more
/// than once, and read in constant time whatever the registry's size; a name two groups spell
/// keeps the first in the registry's order, as the scan it replaces did.
static REPOSITORIES: LazyLock<HashMap<Caseless<'static>, usize>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    for (index, group) in GROUPS.iter().enumerate() {
        for own in group.repositories() {
            map.entry(Caseless(own)).or_insert(index);
        }
    }
    map
});

/// A repository name as a map key, equal to another spelled in any ASCII case: hashed over its
/// ASCII-lowercased bytes, compared with `eq_ignore_ascii_case`.
#[derive(Clone, Copy, Debug)]
struct Caseless<'a>(&'a str);

impl PartialEq for Caseless<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_ignore_ascii_case(other.0)
    }
}

impl Eq for Caseless<'_> {}

impl Hash for Caseless<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Folded into a stack buffer and written once per piece: a hasher's per-byte writes
        // cost more than the fold; the terminator is `str`'s own.
        let mut buffer = [0u8; 64];
        for chunk in self.0.as_bytes().chunks(buffer.len()) {
            let folded = &mut buffer[..chunk.len()];
            folded.copy_from_slice(chunk);
            folded.make_ascii_lowercase();
            state.write(folded);
        }
        state.write_u8(0xff);
    }
}

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
    let repositories: &HashMap<Caseless<'_>, usize> = &REPOSITORIES;
    repositories
        .get(&Caseless(repository))
        .map(|&index| (index, &GROUPS[index]))
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
                    r#"  {{"model": {}, "also": [{}], "table": {}, "opener": {}, "after": {}}}"#,
                    text(group.model),
                    also.join(", "),
                    text(group.table.name()),
                    group.opener.map_or_else(|| "null".to_string(), text),
                    group.after.map_or_else(|| "null".to_string(), text),
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
        // The Llama 3.1 and 3.3 Instruct checkpoints write that syntax under their own ids, the 3.3
        // template byte for byte the 3.1 one: the json table, replayed from the Llama 3 header as
        // the 3.2 rows are. The 3.1 repositories' former names, still served, are listed too.
        for llama in [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "llama-3.3-70b-instruct",
        ] {
            assert_eq!(table_for(llama), Some(Table::Json), "{llama}");
            let format = for_model(llama).unwrap_or_else(|| panic!("{llama}"));
            assert_eq!(
                format.turn_opener(),
                Some("<|start_header_id|>assistant<|end_header_id|>"),
                "{llama}"
            );
        }
        assert_eq!(
            table_for("thinkingmachines/Inkling-NVFP4"),
            Some(Table::Inkling)
        );
        assert_eq!(
            table_for("deepseek-ai/DeepSeek-V4.1-Flash-1017"),
            Some(Table::DeepSeekV4_1)
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
        // The R1 line writes V3's call spelling from its history and a thought the V3 table reads.
        for r1 in [
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-R1-Zero",
            "arcee-ai/DeepSeek-R1-bf16",
            "deepseek-ai/DeepSeek-R1-0528",
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ] {
            assert_eq!(table_for(r1), Some(Table::DeepSeekV3), "{r1}");
        }
    }

    #[test]
    fn a_model_the_registry_does_not_know_resolves_to_none() {
        for model in [
            "Qwen/QwQ-32B",
            "Qwen/Qwen3-8B-FP8",
            // A base checkpoint without the Instruct template: not a name a group lists.
            "meta-llama/Llama-3.1-8B",
            "Qwen",
            "Qwen/",
            "",
            "/",
        ] {
            assert_eq!(table_for(model), None, "{model}");
            assert!(for_model(model).is_none(), "{model}");
        }
    }

    /// The index answers as the scan it replaced did: every checkpoint name of every group, as
    /// the registry spells it and in upper and lower case, finds the first group in the
    /// registry's order that spells it; a name no group spells finds nothing.
    #[test]
    fn every_checkpoint_name_is_found_by_the_index_as_the_scan_found_it_in_any_case() {
        fn scanned(repository: &str) -> Option<usize> {
            GROUPS.iter().position(|group| {
                group
                    .repositories()
                    .any(|own| own.eq_ignore_ascii_case(repository))
            })
        }
        for group in GROUPS {
            for own in group.repositories() {
                let spellings = [
                    own.to_string(),
                    own.to_ascii_uppercase(),
                    own.to_ascii_lowercase(),
                ];
                for spelling in &spellings {
                    let found = group_for(spelling).map(|(index, _)| index);
                    assert_eq!(found, scanned(own), "{spelling}");
                    assert!(found.is_some(), "{spelling}");
                }
            }
        }
        assert_eq!(
            group_for("no-such-checkpoint").map(|(index, _)| index),
            None
        );
        assert_eq!(
            group_for("NO-SUCH-CHECKPOINT").map(|(index, _)| index),
            None
        );
    }

    /// Two spellings of one name are one key: equal, and hashed the same.
    #[test]
    fn a_caseless_key_is_equal_and_hashes_equal_across_ascii_case() {
        use std::hash::{BuildHasher, RandomState};

        let state = RandomState::new();
        let (upper, lower) = (Caseless("Qwen/QWEN3-8B"), Caseless("qwen/qwen3-8b"));
        assert_eq!(upper, lower);
        assert_eq!(state.hash_one(upper), state.hash_one(lower));
        assert_ne!(Caseless("qwen3-8b"), Caseless("qwen3-8"));
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

    /// What a table writes after the thought's close, where it has a thought layout.
    fn after_the_close(format: &Format) -> Option<&str> {
        format
            .layout()
            .and_then(|layout| layout.thought.as_ref())
            .map(|thought| &*thought.after)
    }

    #[test]
    fn every_table_builds_and_takes_its_groups_opener_and_text_after_the_close() {
        for group in GROUPS {
            let format = for_model(group.model).unwrap_or_else(|| panic!("{}", group.model));
            assert_eq!(format.validate(), Ok(()), "{}", group.model);
            let own = group.table.format();
            let expected = group.opener.or_else(|| own.turn_opener());
            assert_eq!(format.turn_opener(), expected, "{}", group.model);
            // A group's own `after` replaces the table's, the close kept; a group with none keeps
            // the table's; a group that names one on a table with no thought layout is refused.
            let expected = group.after.or_else(|| after_the_close(&own));
            assert_eq!(
                after_the_close(&format),
                expected,
                "{}: the text after the thought's close",
                group.model
            );
            if group.after.is_some() {
                assert_eq!(
                    format
                        .layout()
                        .and_then(|layout| layout.thought.as_ref())
                        .and_then(|thought| thought.close.as_deref()),
                    own.layout()
                        .and_then(|layout| layout.thought.as_ref())
                        .and_then(|thought| thought.close.as_deref()),
                    "{}: the close is the table's",
                    group.model
                );
            }
        }
    }

    /// The NVIDIA Nemotron 3 groups and Step 3.5 Flash read with the Qwen 3.5 table, whose template
    /// writes two newlines after the thought's close before a call; theirs write one (bellwether's rendered
    /// `tools-call-arguments-object` for the group: `</think>\n<tool_call>`), which their entries
    /// carry. A forced call written after the close begins with the one newline, and their
    /// thinking-off prompt ends with the bare `<think></think>`, so the newline stays the tag's,
    /// as for the table.
    #[test]
    fn a_nemotron_3_group_writes_its_templates_one_newline_after_the_thoughts_close() {
        use openai_protocol::common::{Function, Tool};
        use serde_json::json;

        use crate::grammar::{GrammarStart, Options};
        let format = for_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16").expect("a table");
        let own = Table::Qwen3Tagged.format();
        assert_eq!(after_the_close(&format), Some("\n"));
        assert_eq!(after_the_close(&own), Some("\n\n"));
        let off = "<|im_start|>assistant\n<think></think>";
        assert!(!format.thought_closed(off));
        assert!(!format.reasoning_open(off));
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                strict: None,
                extra: Default::default(),
            },
        }];
        for (format, newline) in [(&format, "\n"), (&own, "\n\n")] {
            let tag = format
                .grammar_for(
                    &tools,
                    Options {
                        at_least_one: true,
                        reasoning_open: false,
                        thought_closed: false,
                        one_call: true,
                        start: GrammarStart::AfterReasoning,
                    },
                )
                .expect("a tag")
                .payload();
            assert_eq!(tag["format"]["type"], "sequence");
            assert_eq!(tag["format"]["elements"][0]["type"], "const_string");
            assert_eq!(tag["format"]["elements"][0]["value"], newline);
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
            // The frame of a message header's table (Inkling's ended message) stays a wrapper in
            // the variant: the space between two messages is not a call's.
            let frame = format.message_frame();
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
                (1..unread.states())
                    .all(|state| unread.emits(state) != Emits::Wrapper || frame[state]),
                "{}: a calls wrapper remains",
                table.name()
            );
        }
    }
}
