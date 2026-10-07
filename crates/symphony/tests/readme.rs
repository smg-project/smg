//! The README's models table, written from [`GROUPS`] and checked here, so it says what the crate
//! holds: every checkpoint group bellwether has a manifest for, the table that reads it, how far
//! it is, and what SMG takes for it today. When the data changes, `SYMPHONY_WRITE_README=1 cargo
//! test -p smg-symphony --test readme` writes the table into `README.md`; without the switch the
//! test fails on a stale table and prints the one to write.

use std::{fmt::Write as _, fs, path::PathBuf};

use symphony::{
    formats::{
        deepseek_v4_1, hy4, iquest, lfm2_5, ling, minimax_m3, olmo3, qwen2_5, qwen3, seed_oss, xlam,
    },
    CallSyntax, Format,
};

/// The switch that writes the table into the README instead of checking it.
const WRITE_ENV: &str = "SYMPHONY_WRITE_README";
/// The markers in `README.md` around the written table.
const BEGIN: &str = "<!-- models: begin -->";
const END: &str = "<!-- models: end -->";
/// Where bellwether keeps a group's manifest and sets.
const BELLWETHER: &str = "https://github.com/smg-project/bellwether/tree/main/fixtures";

/// A checkpoint group of bellwether's: the checkpoints that share a tokenizer and a template,
/// recorded once by the group's primary.
struct Group {
    /// bellwether's slug for the primary.
    slug: &'static str,
    /// The primary's Hugging Face id.
    model: &'static str,
    /// The group's other checkpoints, by name, under the primary's organisation.
    also: &'static [&'static str],
    /// When the primary was released: the day its Hugging Face repository was created.
    released: &'static str,
    /// What SMG resolves for the primary's id today, before Symphony is wired in: the
    /// `--tool-call-parser` and `--reasoning-parser` names its two registries pick
    /// (`crates/tool_parser`, `crates/reasoning_parser`, read at main 0d0808f2); none is
    /// passthrough, the output read as content.
    smg: (Option<&'static str>, Option<&'static str>),
    /// The table under `formats` that reads the group's output, once one is written.
    table: Option<Table>,
    /// The parse cases bellwether's main holds for the group, once its set is recorded.
    set: Option<u32>,
    /// Where the group stands.
    status: Status,
}

/// A table under `formats`, by the function that builds it.
#[derive(Clone, Copy)]
enum Table {
    Qwen3,
    Qwen3Tagged,
    Qwen2_5,
    DeepSeekV4_1,
    SeedOss,
    Hy4,
    Ling,
    IQuest,
    Olmo3,
    Lfm2_5,
    Xlam,
    MinimaxM3,
}

impl Table {
    /// The table, built; a name in [`GROUPS`] that the crate does not have fails to compile.
    fn format(self) -> Format {
        match self {
            Self::Qwen3 => qwen3(CallSyntax::Json),
            Self::Qwen3Tagged => qwen3(CallSyntax::Tagged),
            Self::Qwen2_5 => qwen2_5(),
            Self::DeepSeekV4_1 => deepseek_v4_1(),
            Self::SeedOss => seed_oss(),
            Self::Hy4 => hy4(),
            Self::Ling => ling(),
            Self::IQuest => iquest(),
            Self::Olmo3 => olmo3(),
            Self::Lfm2_5 => lfm2_5(),
            Self::Xlam => xlam(),
            Self::MinimaxM3 => minimax_m3(),
        }
    }

    /// How the README names it: the function, and the call syntax where the function takes one.
    fn cell(self) -> String {
        match self {
            Self::Qwen3Tagged => format!("`{}`, tagged calls", self.format().name()),
            other => format!("`{}`", other.format().name()),
        }
    }
}

/// Where a group stands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    /// The table is on main, bellwether's benchmark-scale set is recorded, and the fixture test
    /// replays every case of it at every chunking with no difference.
    Ready,
    /// The table is in the open pull request named.
    InReview { pr: u32 },
    /// The table is on main and bellwether's set is recorded; the fixture test has not yet
    /// replayed the whole of it.
    Replaying,
    /// The table is on main; bellwether has not recorded the group's set yet.
    AwaitingFixtures,
    /// No table yet.
    Pending,
}

impl Status {
    fn cell(self) -> String {
        match self {
            Self::Ready => "ready".to_string(),
            Self::InReview { pr } => {
                format!("in review ([#{pr}](https://github.com/smg-project/smg/pull/{pr}))")
            }
            Self::Replaying => "replaying the set".to_string(),
            Self::AwaitingFixtures => "awaiting fixtures".to_string(),
            Self::Pending => "pending".to_string(),
        }
    }
}

/// Every group bellwether has a manifest for, in its slugs' order.
const GROUPS: &[Group] = &[
    Group {
        slug: "ai21-jamba2-3b",
        model: "ai21labs/AI21-Jamba2-3B",
        also: &[],
        released: "2026-01-06",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "apertus-8b-instruct-2509",
        model: "swiss-ai/Apertus-8B-Instruct-2509",
        also: &[],
        released: "2025-08-13",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-r1",
        model: "deepseek-ai/DeepSeek-R1",
        also: &[],
        released: "2025-01-20",
        smg: (Some("pythonic"), Some("deepseek_r1")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-v3-0324",
        model: "deepseek-ai/DeepSeek-V3-0324",
        also: &[],
        released: "2025-03-24",
        smg: (Some("deepseek"), None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-v3.1",
        model: "deepseek-ai/DeepSeek-V3.1",
        also: &[],
        released: "2025-08-21",
        smg: (Some("deepseek31"), Some("deepseek_v31")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-v4.1-flash",
        model: "deepseek-ai/DeepSeek-V4.1-Flash",
        also: &[],
        released: "2026-09-10",
        smg: (Some("deepseek_v41"), Some("deepseek_v41")),
        table: Some(Table::DeepSeekV4_1),
        set: Some(54_418),
        status: Status::Replaying,
    },
    Group {
        slug: "dots3-note-prev",
        model: "dots-studio/dots3-note-prev",
        also: &[],
        released: "2026-08-09",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "ernie-4.5-21b-a3b-thinking",
        model: "baidu/ERNIE-4.5-21B-A3B-Thinking",
        also: &[],
        released: "2025-09-08",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "gemma-4-e4b-it",
        model: "google/gemma-4-E4B-it",
        also: &[],
        released: "2026-03-02",
        smg: (Some("json"), None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "glm-4.6",
        model: "zai-org/GLM-4.6",
        also: &[],
        released: "2025-09-29",
        smg: (Some("glm45_moe"), None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "glm-4.7-flash",
        model: "zai-org/GLM-4.7-Flash",
        also: &[],
        released: "2026-01-19",
        smg: (Some("glm47_moe"), None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "glm-5.3-flash",
        model: "zai-org/GLM-5.3-Flash",
        also: &[],
        released: "2026-08-25",
        smg: (Some("glm47_moe"), Some("glm45")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "granite-4.1-3b",
        model: "ibm-granite/granite-4.1-3b",
        also: &[],
        released: "2026-04-06",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "hermes-4-14b",
        model: "NousResearch/Hermes-4-14B",
        also: &[],
        released: "2025-08-30",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "hunyuan-a13b-instruct",
        model: "tencent/Hunyuan-A13B-Instruct",
        also: &[],
        released: "2025-06-25",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "hy4-preview",
        model: "tencent/Hy4-preview",
        also: &[],
        released: "2026-08-27",
        smg: (Some("hy_v4"), Some("hy_v4")),
        table: Some(Table::Hy4),
        set: Some(54_195),
        status: Status::Replaying,
    },
    Group {
        slug: "inkling",
        model: "thinkingmachines/Inkling",
        also: &[],
        released: "2026-07-14",
        smg: (Some("inkling"), Some("inkling")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "iquest-q1",
        model: "IQuestLab/IQuest-Q1",
        also: &[],
        released: "2026-09-28",
        smg: (None, None),
        table: Some(Table::IQuest),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "k-exaone-236b-a23b",
        model: "LGAI-EXAONE/K-EXAONE-236B-A23B",
        also: &[],
        released: "2025-12-26",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "k2-horizon-36b",
        model: "IFM/K2-Horizon-36B",
        also: &[],
        released: "2026-09-01",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "laguna-xs.2",
        model: "poolside/Laguna-XS.2",
        also: &[],
        released: "2026-04-23",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "lfm2.5-1.2b-instruct",
        model: "LiquidAI/LFM2.5-1.2B-Instruct",
        also: &[],
        released: "2026-01-06",
        smg: (None, None),
        table: Some(Table::Lfm2_5),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "ling-3.0-flash",
        model: "inclusionAI/Ling-3.0-flash",
        also: &[],
        released: "2026-08-02",
        smg: (None, None),
        table: Some(Table::Ling),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "llama-xlam-2-8b-fc-r",
        model: "Salesforce/Llama-xLAM-2-8b-fc-r",
        also: &[],
        released: "2025-03-27",
        smg: (Some("json"), None),
        table: Some(Table::Xlam),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "llava-1.5-7b-hf",
        model: "llava-hf/llava-1.5-7b-hf",
        also: &[],
        released: "2023-12-05",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "mimo-v2.5",
        model: "XiaomiMiMo/MiMo-V2.5",
        also: &[],
        released: "2026-04-27",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "minicpm5-2b",
        model: "openbmb/MiniCPM5-2B",
        also: &[],
        released: "2026-09-06",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "minimax-m2",
        model: "MiniMaxAI/MiniMax-M2",
        also: &[],
        released: "2025-10-22",
        smg: (Some("minimax_m2"), Some("minimax")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "minimax-m2.7",
        model: "MiniMaxAI/MiniMax-M2.7",
        also: &[],
        released: "2026-04-09",
        smg: (Some("minimax_m2"), Some("minimax")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "minimax-m3",
        model: "MiniMaxAI/MiniMax-M3",
        also: &[],
        released: "2026-06-02",
        smg: (Some("minimax_m3"), Some("minimax_m3")),
        table: Some(Table::MinimaxM3),
        set: Some(54_180),
        status: Status::InReview { pr: 2850 },
    },
    Group {
        slug: "mistral-7b-instruct-v0.3",
        model: "mistralai/Mistral-7B-Instruct-v0.3",
        also: &[],
        released: "2024-05-22",
        smg: (Some("mistral"), None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "muse-glimmer-30b",
        model: "meta-models/Muse-Glimmer-30B",
        also: &[],
        released: "2026-08-09",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "nanbeige4.2-3b",
        model: "Nanbeige/Nanbeige4.2-3B",
        also: &[],
        released: "2026-07-21",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "nvidia-nemotron-3-nano-30b-a3b-bf16",
        model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        also: &[],
        released: "2025-12-04",
        smg: (Some("qwen_xml"), Some("nano_v3")),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "olmo-3-7b-instruct",
        model: "allenai/Olmo-3-7B-Instruct",
        also: &[],
        released: "2025-11-19",
        smg: (None, None),
        table: Some(Table::Olmo3),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "phi-4-mini-instruct",
        model: "microsoft/Phi-4-mini-instruct",
        also: &[],
        released: "2025-02-19",
        smg: (None, None),
        table: None,
        set: Some(27_865),
        status: Status::Pending,
    },
    Group {
        slug: "phi-4-multimodal-instruct",
        model: "microsoft/Phi-4-multimodal-instruct",
        also: &[],
        released: "2025-02-24",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "qwen-agentworld-35b-a3b",
        model: "Qwen/Qwen-AgentWorld-35B-A3B",
        also: &[],
        released: "2026-06-22",
        smg: (Some("qwen"), Some("qwen3")),
        table: None,
        set: Some(54_195),
        status: Status::Pending,
    },
    Group {
        slug: "qwen-drive-1.0-4b",
        model: "Qwen/Qwen-Drive-1.0-4B",
        also: &[],
        released: "2026-08-27",
        smg: (Some("qwen"), Some("qwen3")),
        table: None,
        set: Some(54_181),
        status: Status::Pending,
    },
    Group {
        slug: "qwen2.5-7b-instruct-1m",
        model: "Qwen/Qwen2.5-7B-Instruct-1M",
        also: &["Qwen2.5-14B-Instruct-1M"],
        released: "2025-01-23",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen2_5),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen2.5-omni-7b",
        model: "Qwen/Qwen2.5-Omni-7B",
        also: &["Qwen2.5-Omni-3B"],
        released: "2025-03-22",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen2_5),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen2.5-vl-32b-instruct",
        model: "Qwen/Qwen2.5-VL-32B-Instruct",
        also: &[],
        released: "2025-03-21",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen2_5),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen2.5-vl-7b-instruct",
        model: "Qwen/Qwen2.5-VL-7B-Instruct",
        also: &["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-72B-Instruct"],
        released: "2025-01-26",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen2_5),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3-30b-a3b",
        model: "Qwen/Qwen3-30B-A3B",
        also: &["Qwen3-235B-A22B"],
        released: "2025-04-27",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-30b-a3b-instruct-2507",
        model: "Qwen/Qwen3-30B-A3B-Instruct-2507",
        also: &["Qwen3-235B-A22B-Instruct-2507"],
        released: "2025-07-28",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(41_533),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-30b-a3b-thinking-2507",
        model: "Qwen/Qwen3-30B-A3B-Thinking-2507",
        also: &["Qwen3-235B-A22B-Thinking-2507"],
        released: "2025-07-29",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-4b-instruct-2507",
        model: "Qwen/Qwen3-4B-Instruct-2507",
        also: &[],
        released: "2025-08-05",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(41_533),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-4b-saferl",
        model: "Qwen/Qwen3-4B-SafeRL",
        also: &[],
        released: "2025-09-30",
        smg: (Some("qwen"), Some("qwen3")),
        table: None,
        set: Some(54_418),
        status: Status::Pending,
    },
    Group {
        slug: "qwen3-4b-thinking-2507",
        model: "Qwen/Qwen3-4B-Thinking-2507",
        also: &[],
        released: "2025-08-05",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-8b",
        model: "Qwen/Qwen3-8B",
        also: &[
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-14B",
            "Qwen3-32B",
        ],
        released: "2025-04-27",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-coder-30b-a3b-instruct",
        model: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        also: &["Qwen3-Coder-480B-A35B-Instruct"],
        released: "2025-07-31",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(41_362),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-coder-next",
        model: "Qwen/Qwen3-Coder-Next",
        also: &[],
        released: "2026-01-30",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(41_374),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-next-80b-a3b-instruct",
        model: "Qwen/Qwen3-Next-80B-A3B-Instruct",
        also: &[],
        released: "2025-09-09",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(41_533),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-next-80b-a3b-thinking",
        model: "Qwen/Qwen3-Next-80B-A3B-Thinking",
        also: &[],
        released: "2025-09-09",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-omni-30b-a3b-instruct",
        model: "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        also: &[],
        released: "2025-09-20",
        smg: (Some("qwen"), Some("qwen3")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "qwen3-omni-30b-a3b-thinking",
        model: "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        also: &[],
        released: "2025-09-15",
        smg: (Some("qwen"), Some("qwen3")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "qwen3-vl-235b-a22b-thinking",
        model: "Qwen/Qwen3-VL-235B-A22B-Thinking",
        also: &["Qwen3-VL-30B-A3B-Thinking"],
        released: "2025-09-22",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3-vl-30b-a3b-instruct",
        model: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        also: &["Qwen3-VL-235B-A22B-Instruct"],
        released: "2025-09-30",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3-vl-8b-instruct",
        model: "Qwen/Qwen3-VL-8B-Instruct",
        also: &[
            "Qwen3-VL-2B-Instruct",
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-32B-Instruct",
        ],
        released: "2025-10-11",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3-vl-8b-thinking",
        model: "Qwen/Qwen3-VL-8B-Thinking",
        also: &[
            "Qwen3-VL-2B-Thinking",
            "Qwen3-VL-4B-Thinking",
            "Qwen3-VL-32B-Thinking",
        ],
        released: "2025-10-11",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3.5-27b",
        model: "Qwen/Qwen3.5-27B",
        also: &[],
        released: "2026-02-24",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3.5-2b",
        model: "Qwen/Qwen3.5-2B",
        also: &["Qwen3.5-0.8B"],
        released: "2026-02-28",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3.5-35b-a3b",
        model: "Qwen/Qwen3.5-35B-A3B",
        also: &["Qwen3.5-122B-A10B", "Qwen3.5-397B-A17B"],
        released: "2026-02-24",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3.5-9b",
        model: "Qwen/Qwen3.5-9B",
        also: &["Qwen3.5-4B"],
        released: "2026-02-27",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3.6-27b",
        model: "Qwen/Qwen3.6-27B",
        also: &[],
        released: "2026-04-21",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3.6-35b-a3b",
        model: "Qwen/Qwen3.6-35B-A3B",
        also: &[],
        released: "2026-04-15",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3.8-2.4t-a95b",
        model: "Qwen/Qwen3.8-2.4T-A95B",
        also: &[],
        released: "2026-08-08",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "qwen3.8-27b",
        model: "Qwen/Qwen3.8-27B",
        also: &[],
        released: "2026-08-05",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "qwen3.8-flash-next",
        model: "Qwen/Qwen3.8-Flash-Next",
        also: &[],
        released: "2026-08-24",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "qwen3guard-gen-0.6b",
        model: "Qwen/Qwen3Guard-Gen-0.6B",
        also: &["Qwen3Guard-Gen-4B", "Qwen3Guard-Gen-8B"],
        released: "2025-09-23",
        smg: (Some("qwen"), Some("qwen3")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "qwq-32b",
        model: "Qwen/QwQ-32B",
        also: &[],
        released: "2025-03-05",
        smg: (Some("qwen"), Some("qwen3")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "seed-oss-36b-instruct",
        model: "ByteDance-Seed/Seed-OSS-36B-Instruct",
        also: &[],
        released: "2025-08-20",
        smg: (None, None),
        table: Some(Table::SeedOss),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "step-3.5-flash",
        model: "stepfun-ai/Step-3.5-Flash",
        also: &[],
        released: "2026-02-01",
        smg: (Some("step3"), None),
        table: Some(Table::Qwen3Tagged),
        set: None,
        status: Status::AwaitingFixtures,
    },
    Group {
        slug: "step3",
        model: "stepfun-ai/step3",
        also: &[],
        released: "2025-07-28",
        smg: (Some("step3"), Some("step3")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "tinyllama-1.1b-chat-v1.0",
        model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        also: &[],
        released: "2023-12-30",
        smg: (Some("json"), None),
        table: None,
        set: Some(27_865),
        status: Status::Pending,
    },
    Group {
        slug: "trinity-mini",
        model: "arcee-ai/Trinity-Mini",
        also: &[],
        released: "2025-12-01",
        smg: (None, None),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "webworld-32b",
        model: "Qwen/WebWorld-32B",
        also: &["WebWorld-8B", "WebWorld-14B"],
        released: "2026-02-13",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: None,
        status: Status::AwaitingFixtures,
    },
];

/// `n` with thousands separators, as the README writes counts.
fn count(n: u32) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The table and the sentence above it, from [`GROUPS`].
fn render() -> String {
    let mut groups: Vec<&Group> = GROUPS.iter().collect();
    groups.sort_by_key(|group| (std::cmp::Reverse(group.released), group.slug));
    let tally = |status: fn(&Status) -> bool| groups.iter().filter(|g| status(&g.status)).count();
    let checkpoints: usize = groups.iter().map(|group| 1 + group.also.len()).sum();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{} groups, {} checkpoints, newest first.\n{} ready, {} in review, {} replaying a recorded \
         set, {} on main awaiting fixtures, {} pending.",
        groups.len(),
        checkpoints,
        tally(|s| *s == Status::Ready),
        tally(|s| matches!(s, Status::InReview { .. })),
        tally(|s| *s == Status::Replaying),
        tally(|s| *s == Status::AwaitingFixtures),
        tally(|s| *s == Status::Pending),
    );
    out.push('\n');
    out.push_str(
        "| Released | Group | Model | Also | Table | bellwether set | Status | SMG today |\n",
    );
    out.push_str("|---|---|---|---|---|---:|---|---|\n");
    for group in groups {
        let smg = match group.smg {
            (None, None) => "none".to_string(),
            (tool, reasoning) => {
                let name =
                    |n: Option<&str>| n.map_or_else(|| "none".to_string(), |n| format!("`{n}`"));
                format!("{}, {}", name(tool), name(reasoning))
            }
        };
        let _ = writeln!(
            out,
            "| {released} | [{slug}]({BELLWETHER}/{slug}) | {model} | {also} | {table} | {set} | \
             {status} | {smg} |",
            released = group.released,
            slug = group.slug,
            model = group.model,
            also = group.also.join(", "),
            table = group.table.map_or_else(|| "—".to_string(), Table::cell),
            set = group.set.map_or_else(|| "—".to_string(), count),
            status = group.status.cell(),
        );
    }
    out
}

fn readme_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("README.md")
}

#[test]
fn the_readme_models_table_is_what_groups_says() {
    let path = readme_path();
    let readme = fs::read_to_string(&path).expect("README.md beside Cargo.toml");
    let begin = readme.find(BEGIN).expect("the begin marker in README.md") + BEGIN.len();
    let end = readme.find(END).expect("the end marker in README.md");
    assert!(begin <= end, "the markers in README.md are out of order");
    let expected = format!("\n{}", render());
    if std::env::var(WRITE_ENV).is_ok_and(|value| value == "1") {
        let written = format!("{}{}{}", &readme[..begin], expected, &readme[end..]);
        fs::write(&path, written).expect("README.md written");
        return;
    }
    assert!(
        readme[begin..end] == expected,
        "README.md's models table is not what GROUPS says; run `{WRITE_ENV}=1 cargo test -p \
         smg-symphony --test readme` to write it, or replace everything between the markers with:\n\
         {expected}"
    );
}

#[test]
fn every_named_table_builds_and_every_status_has_what_it_claims() {
    for group in GROUPS {
        if let Some(table) = group.table {
            table
                .format()
                .validate()
                .unwrap_or_else(|why| panic!("{}: {why}", group.slug));
        }
        // Every status but pending claims a table; ready and replaying claim a recorded set too.
        assert_eq!(
            group.table.is_some(),
            group.status != Status::Pending,
            "{}: a table and a status that disagree",
            group.slug
        );
        if matches!(group.status, Status::Ready | Status::Replaying) {
            assert!(group.set.is_some(), "{} has no recorded set", group.slug);
        }
        if group.status == Status::AwaitingFixtures {
            assert!(
                group.set.is_none(),
                "{} awaits fixtures with a recorded set",
                group.slug
            );
        }
    }
}

/// Whether `date` is a calendar day written `YYYY-MM-DD`, from 2020 on: the month in its range,
/// the day in the month's, February's 29th in a leap year only.
fn is_a_day(date: &str) -> bool {
    let Some((year, rest)) = date.split_once('-') else {
        return false;
    };
    let Some((month, day)) = rest.split_once('-') else {
        return false;
    };
    if year.len() != 4
        || month.len() != 2
        || day.len() != 2
        || ![year, month, day]
            .iter()
            .all(|part| part.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (
        year.parse::<u32>(),
        month.parse::<u32>(),
        day.parse::<u32>(),
    ) else {
        return false;
    };
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year >= 2020 && (1..=days).contains(&day)
}

#[test]
fn every_release_date_is_a_calendar_day() {
    for group in GROUPS {
        assert!(
            is_a_day(group.released),
            "{}: released {:?} is not a day written YYYY-MM-DD",
            group.slug,
            group.released
        );
    }
    for not_a_day in [
        "2026-02-30",
        "2025-13-01",
        "2024-04-31",
        "2019-12-31",
        "2026-9-1",
        "2026-+9-01",
        "today",
    ] {
        assert!(!is_a_day(not_a_day), "{not_a_day:?} is not a day");
    }
    assert!(is_a_day("2024-02-29") && !is_a_day("2023-02-29"));
}

#[test]
fn groups_are_unique_and_in_slug_order() {
    let slugs: Vec<&str> = GROUPS.iter().map(|group| group.slug).collect();
    let mut sorted = slugs.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(slugs, sorted, "GROUPS is listed by slug, each once");
}

#[test]
fn counts_are_written_with_separators() {
    assert_eq!(count(7), "7");
    assert_eq!(count(999), "999");
    assert_eq!(count(1_000), "1,000");
    assert_eq!(count(54_418), "54,418");
    assert_eq!(count(1_234_567), "1,234,567");
}
