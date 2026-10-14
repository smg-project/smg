//! The README's models table, written from [`GROUPS`] and checked here, so it says what the crate
//! holds: every checkpoint group bellwether has a manifest for, the table that reads it, how far
//! it is, and what SMG takes for it today. When the data changes, `SYMPHONY_WRITE_README=1 cargo
//! test -p smg-symphony --test readme` writes the table into `README.md`; without the switch the
//! test fails on a stale table and prints the one to write.

use std::{fmt::Write as _, fs, path::PathBuf};

use openai_protocol::common::{Function, Tool};
use serde_json::json;
use symphony::formats::{groups, table_for, Table};

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
    /// The table under `formats` that reads the group's output, once one is written; the
    /// registry under `formats` must say the same for the primary and every `also`.
    table: Option<Table>,
    /// The parse cases bellwether's main holds for the group, once its set is recorded.
    set: Option<u32>,
    /// Where the group stands.
    status: Status,
}

/// How the README names a table: the function, and the call syntax where the function takes one.
fn table_cell(table: Table) -> String {
    match table {
        Table::Qwen3Tagged => format!("`{}`, tagged calls", table.format().name()),
        Table::Qwen3TaggedDirect => format!(
            "`{}`, tagged calls, the value between the tags",
            table.format().name()
        ),
        other => format!("`{}`", other.format().name()),
    }
}

/// Where a group stands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    /// The table is on main, bellwether's benchmark-scale set is recorded, and a local run of the
    /// fixture test on main's code replayed every case of it at every chunking with no
    /// difference; CI guards the set once its bellwether pin holds it.
    Ready,
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
            Self::Replaying => "replaying the set".to_string(),
            Self::AwaitingFixtures => "awaiting fixtures".to_string(),
            Self::Pending => "pending".to_string(),
        }
    }
}

/// Every group bellwether has recorded parse sets for, and the groups listed here before its
/// manifests outgrew the table, in slug order.
const GROUPS: &[Group] = &[
    Group {
        slug: "a.x-3.1-light",
        model: "skt/A.X-3.1-Light",
        also: &[],
        released: "2025-07-10",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "a.x-4.0",
        model: "skt/A.X-4.0",
        also: &[],
        released: "2025-07-02",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "a.x-4.0-light",
        model: "skt/A.X-4.0-Light",
        also: &[],
        released: "2025-07-02",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "academic-ds-9b",
        model: "ByteDance-Seed/academic-ds-9B",
        also: &[],
        released: "2025-04-09",
        smg: (None, None),
        table: None,
        set: Some(3_114),
        status: Status::Pending,
    },
    Group {
        slug: "acereason-nemotron-7b",
        model: "nvidia/AceReason-Nemotron-7B",
        also: &[
            "AceMath-RL-Nemotron-7B",
            "AceReason-Nemotron-14B",
            "Nemotron-Research-Reasoning-Qwen-1.5B",
        ],
        released: "2025-05-22",
        smg: (None, None),
        table: None,
        set: Some(94_830),
        status: Status::Pending,
    },
    Group {
        slug: "afm-4.5b",
        model: "arcee-ai/AFM-4.5B",
        also: &[],
        released: "2025-07-29",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "agentcpm-explore",
        model: "openbmb/AgentCPM-Explore",
        also: &[],
        released: "2026-01-11",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "agentcpm-report",
        model: "openbmb/AgentCPM-Report",
        also: &[],
        released: "2026-01-19",
        smg: (None, None),
        table: None,
        set: Some(328_660),
        status: Status::Pending,
    },
    Group {
        slug: "ai21-jamba-reasoning-3b",
        model: "ai21labs/AI21-Jamba-Reasoning-3B",
        also: &[],
        released: "2025-10-05",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(12_885),
        status: Status::Replaying,
    },
    Group {
        slug: "ai21-jamba2-3b",
        model: "ai21labs/AI21-Jamba2-3B",
        also: &["AI21-Jamba2-Mini"],
        released: "2026-01-06",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "apertus-70b-instruct-2509",
        model: "swiss-ai/Apertus-70B-Instruct-2509",
        also: &[],
        released: "2025-09-01",
        smg: (None, None),
        table: None,
        set: Some(137_178),
        status: Status::Pending,
    },
    Group {
        slug: "apertus-8b-instruct-2509",
        model: "swiss-ai/Apertus-8B-Instruct-2509",
        also: &[],
        released: "2025-08-13",
        smg: (None, None),
        table: None,
        set: Some(137_178),
        status: Status::Pending,
    },
    Group {
        slug: "apertus-v1.1-4b-instruct",
        model: "swiss-ai/Apertus-v1.1-4B-Instruct",
        also: &["Apertus-v1.1-0.5B-Instruct", "Apertus-v1.1-1.5B-Instruct"],
        released: "2026-05-01",
        smg: (None, None),
        table: None,
        set: Some(184_972),
        status: Status::Pending,
    },
    Group {
        slug: "areal-boba-2-14b",
        model: "inclusionAI/AReaL-boba-2-14B",
        also: &[],
        released: "2025-06-03",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "areal-boba-2-32b",
        model: "inclusionAI/AReaL-boba-2-32B",
        also: &[],
        released: "2025-06-03",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "areal-boba-2-8b",
        model: "inclusionAI/AReaL-boba-2-8B",
        also: &["AReaL-boba-2-14B-Open", "AReaL-boba-2-8B-Open"],
        released: "2025-06-03",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "areal-sea-235b-a22b",
        model: "inclusionAI/AReaL-SEA-235B-A22B",
        also: &["AReaL-tau2-merge-sft-235B"],
        released: "2026-03-01",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(554_276),
        status: Status::Replaying,
    },
    Group {
        slug: "areal-tau2-telecom-sft-30b",
        model: "inclusionAI/AReaL-tau2-telecom-sft-30B",
        also: &["AReaL-tau2-airline-sft-30B", "AReaL-tau2-retail-sft-30B"],
        released: "2026-03-05",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(554_276),
        status: Status::Replaying,
    },
    Group {
        slug: "areal-tau2-telecom-user-sft-30b",
        model: "inclusionAI/AReaL-tau2-telecom-user-sft-30B",
        also: &[],
        released: "2026-03-05",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "arex-2",
        model: "BAAI/AREX-2",
        also: &[],
        released: "2026-09-29",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_563),
        status: Status::Replaying,
    },
    Group {
        slug: "arex-base",
        model: "BAAI/AREX-Base",
        also: &[],
        released: "2026-07-23",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_543),
        status: Status::Replaying,
    },
    Group {
        slug: "arex-turbo",
        model: "BAAI/AREX-Turbo",
        also: &[],
        released: "2026-07-23",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_543),
        status: Status::Replaying,
    },
    Group {
        slug: "astabrief-8b",
        model: "allenai/AstaBrief_8B",
        also: &["AstaBrief_8B_SFT"],
        released: "2026-02-09",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "atria-dawn-preview",
        model: "internlm/Atria-Dawn-Preview",
        also: &["Atria-Dawn-Preview-Ascend-w8a8"],
        released: "2026-09-11",
        smg: (None, None),
        table: Some(Table::Glm),
        set: Some(503_273),
        status: Status::Replaying,
    },
    Group {
        slug: "audio-flamingo-3-hf",
        model: "nvidia/audio-flamingo-3-hf",
        also: &[],
        released: "2025-10-24",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "autoglm-phone-9b",
        model: "zai-org/AutoGLM-Phone-9B",
        also: &[],
        released: "2025-12-08",
        smg: (Some("json"), None),
        table: None,
        set: Some(328_664),
        status: Status::Pending,
    },
    Group {
        slug: "autoglm-phone-9b-multilingual",
        model: "zai-org/AutoGLM-Phone-9B-Multilingual",
        also: &[],
        released: "2025-12-09",
        smg: (Some("json"), None),
        table: None,
        set: Some(328_664),
        status: Status::Pending,
    },
    Group {
        slug: "baichuan-m3-235b",
        model: "baichuan-inc/Baichuan-M3-235B",
        also: &[],
        released: "2026-01-13",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(554_276),
        status: Status::Replaying,
    },
    Group {
        slug: "baichuanmed-ocr-7b",
        model: "baichuan-inc/BaichuanMed-OCR-7B",
        also: &["BaichuanMed-OCR-72B"],
        released: "2025-04-07",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "bar-2x7b-base",
        model: "allenai/BAR-2x7B-Base",
        also: &[],
        released: "2026-04-19",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "bar-2x7b-code",
        model: "allenai/BAR-2x7B-Code",
        also: &[
            "BAR-2x7B-Code-SFT",
            "BAR-2x7B-Safety",
            "BAR-2x7B-Tool-Use",
            "BAR-5x7B",
        ],
        released: "2026-04-19",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "bar-2x7b-math",
        model: "allenai/BAR-2x7B-Math",
        also: &["BAR-2x7B-Math-SFT"],
        released: "2026-04-19",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "bar-7b",
        model: "allenai/BAR-7B",
        also: &[],
        released: "2026-04-19",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "bitcpm-cann-3b-unquantized",
        model: "openbmb/BitCPM-CANN-3B-unquantized",
        also: &[
            "BitCPM-CANN-1B",
            "BitCPM-CANN-1B-unquantized",
            "BitCPM-CANN-3B",
        ],
        released: "2026-05-18",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "bytedance-seed-oss-36b-alternate-tokenizer",
        model: "NousResearch/ByteDance-Seed-OSS-36B-Alternate-Tokenizer",
        also: &[],
        released: "2025-09-10",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "caller",
        model: "arcee-ai/Caller",
        also: &[],
        released: "2025-06-09",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(324_160),
        status: Status::Replaying,
    },
    Group {
        slug: "caprl-3b",
        model: "internlm/CapRL-3B",
        also: &[],
        released: "2025-09-24",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "caprl-eval-3b",
        model: "internlm/CapRL-Eval-3B",
        also: &[],
        released: "2025-09-25",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "caprl-internvl3.5-8b",
        model: "internlm/CapRL-InternVL3.5-8B",
        also: &[],
        released: "2025-10-16",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "caprl-qwen3vl-4b",
        model: "internlm/CapRL-Qwen3VL-4B",
        also: &["CapRL-Qwen3VL-2B"],
        released: "2025-12-24",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "caprl-video-4b",
        model: "internlm/CapRL-Video-4B",
        also: &["inclusionAI/ArmorOCR"],
        released: "2026-05-22",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "cogalign-internvl2-5-mpo-4b",
        model: "Salesforce/cogalign-internvl2_5-mpo-4b",
        also: &["cogalign-internvl2_5-mpo-1b"],
        released: "2025-02-17",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "cogalign-llava-ov-0-5b",
        model: "Salesforce/cogalign-llava-ov-0_5b",
        also: &[],
        released: "2025-02-17",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "contextpilot-14b",
        model: "tencent/ContextPilot-14B",
        also: &[
            "ByteDance-Seed/cudaLLM-8B",
            "FreedomIntelligence/HuatuoGPT-3-32B",
            "nvidia/Nemotron-Terminal-14B",
            "nvidia/Nemotron-Terminal-32B",
            "nvidia/Nemotron-Terminal-8B",
            "OpenMOSS-Team/Qwen3-32B-ABC",
            "ContextPilot-8B",
        ],
        released: "2026-08-27",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "cosmos-reason1-7b",
        model: "nvidia/Cosmos-Reason1-7B",
        also: &[],
        released: "2025-04-18",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "cosmos3-edge",
        model: "nvidia/Cosmos3-Edge",
        also: &[],
        released: "2026-07-01",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "cosmos3-edge-policy-droid",
        model: "nvidia/Cosmos3-Edge-Policy-DROID",
        also: &[],
        released: "2026-07-09",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "crispers-7b-v1",
        model: "thu-coai/Crispers-7B-v1",
        also: &["Crispers-14B-v1"],
        released: "2025-04-24",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "d1-3b",
        model: "LiquidAI/d1-3B",
        also: &[],
        released: "2026-10-05",
        smg: (None, None),
        table: Some(Table::Lfm2_5),
        set: Some(589_657),
        status: Status::Replaying,
    },
    Group {
        slug: "deep-ignorance-unfiltered-instruct-test",
        model: "EleutherAI/deep-ignorance-unfiltered-instruct-test",
        also: &[
            "deep-ignorance-e2e-strong-filter-instruct-test",
            "deep-ignorance-e2e-strong-filter-instruct-test-v2",
            "deep-ignorance-strong-filter-pt-weak-filter-anneal-instruct-test",
            "deep-ignorance-strong-filter-pt-weak-filter-anneal-instruct-test-v2",
        ],
        released: "2025-10-13",
        smg: (None, None),
        table: None,
        set: Some(273_678),
        status: Status::Pending,
    },
    Group {
        slug: "deephermes-financial-fundamentals-prediction-specialist-atropos",
        model: "NousResearch/DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos",
        also: &[
            "DeepHermes-AscensionMaze-RLAIF-8b-Atropos",
            "DeepHermes-Egregore-v1-RLAIF-8b-Atropos",
            "DeepHermes-Egregore-v2-RLAIF-8b-Atropos",
            "DeepHermes-ToolCalling-Specialist-Atropos",
        ],
        released: "2025-04-28",
        smg: (None, None),
        table: None,
        set: Some(137_178),
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-r1",
        model: "deepseek-ai/DeepSeek-R1",
        also: &["DeepSeek-R1-Zero"],
        released: "2025-01-20",
        smg: (Some("pythonic"), Some("deepseek_r1")),
        table: None,
        set: None,
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-r1-bf16",
        model: "arcee-ai/DeepSeek-R1-bf16",
        also: &[],
        released: "2025-01-20",
        smg: (Some("pythonic"), Some("deepseek_r1")),
        table: None,
        set: Some(3_110),
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-v3-0324",
        model: "deepseek-ai/DeepSeek-V3-0324",
        also: &["DeepSeek-Prover-V2-671B"],
        released: "2025-03-24",
        smg: (Some("deepseek"), None),
        table: None,
        set: Some(19_299),
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-v3-0324-bf16",
        model: "arcee-ai/DeepSeek-V3-0324-bf16",
        also: &[],
        released: "2025-03-24",
        smg: (Some("deepseek"), None),
        table: None,
        set: Some(3_110),
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-v3.1",
        model: "deepseek-ai/DeepSeek-V3.1",
        also: &[
            "DeepSeek-V3.1-Base",
            "tencent/DeepSeek-V3.1-Terminus-W4AFP8",
        ],
        released: "2025-08-21",
        smg: (Some("deepseek31"), Some("deepseek_v31")),
        table: None,
        set: Some(19_299),
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-v3.2-exp",
        model: "deepseek-ai/DeepSeek-V3.2-Exp",
        also: &["DeepSeek-V3.2-Exp-Base"],
        released: "2025-09-29",
        smg: (Some("deepseek31"), None),
        table: None,
        set: Some(184_972),
        status: Status::Pending,
    },
    Group {
        slug: "deepseek-v4.1-flash",
        model: "deepseek-ai/DeepSeek-V4.1-Flash",
        also: &[],
        released: "2026-09-10",
        smg: (Some("deepseek_v41"), Some("deepseek_v41")),
        table: Some(Table::DeepSeekV4_1),
        set: Some(589_659),
        status: Status::Replaying,
    },
    Group {
        slug: "devstral-2-123b-instruct-2512",
        model: "mistralai/Devstral-2-123B-Instruct-2512",
        also: &[],
        released: "2025-11-28",
        smg: (None, None),
        table: Some(Table::Mistral),
        set: Some(570_194),
        status: Status::Replaying,
    },
    Group {
        slug: "dler-llama-nemotron-8b-merge-research",
        model: "nvidia/DLER-Llama-Nemotron-8B-Merge-Research",
        also: &[],
        released: "2025-08-11",
        smg: (Some("json"), None),
        table: Some(Table::LlamaNemotron),
        set: Some(416_850),
        status: Status::Replaying,
    },
    Group {
        slug: "dler-r1-1.5b-research",
        model: "nvidia/DLER-R1-1.5B-Research",
        also: &["DLER-R1-7B-Research"],
        released: "2025-08-11",
        smg: (None, None),
        table: None,
        set: Some(94_830),
        status: Status::Pending,
    },
    Group {
        slug: "dogr",
        model: "tencent/DOGR",
        also: &[],
        released: "2025-08-06",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "dots.mocr",
        model: "dots-studio/dots.mocr",
        also: &["dots.mocr-svg"],
        released: "2026-03-19",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "dots3-note-prev",
        model: "dots-studio/dots3-note-prev",
        also: &[],
        released: "2026-08-09",
        smg: (None, None),
        table: Some(Table::Dots3),
        set: Some(589_296),
        status: Status::Replaying,
    },
    Group {
        slug: "e1-code-14b",
        model: "Salesforce/E1-Code-14B",
        also: &[],
        released: "2025-05-19",
        smg: (None, None),
        table: None,
        set: Some(94_830),
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
        slug: "ernie-image-aes",
        model: "baidu/ERNIE-Image-Aes",
        also: &[],
        released: "2026-05-18",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "exaone-4.5-33b",
        model: "LGAI-EXAONE/EXAONE-4.5-33B",
        also: &[],
        released: "2026-04-04",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(503_634),
        status: Status::Replaying,
    },
    Group {
        slug: "falcon-h1-tiny-90m-instruct",
        model: "tiiuae/Falcon-H1-Tiny-90M-Instruct",
        also: &["Falcon-H1-Tiny-90M-Instruct-Curriculum"],
        released: "2026-01-12",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(429_360),
        status: Status::Replaying,
    },
    Group {
        slug: "falcon-h1-tiny-multilingual-100m-instruct",
        model: "tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct",
        also: &[],
        released: "2026-01-12",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "falcon-h1-tiny-r-0.6b",
        model: "tiiuae/Falcon-H1-Tiny-R-0.6B",
        also: &[],
        released: "2026-01-12",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "falcon-h1-tiny-r-90m",
        model: "tiiuae/Falcon-H1-Tiny-R-90M",
        also: &[],
        released: "2026-01-12",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "falcon-h1r-7b",
        model: "tiiuae/Falcon-H1R-7B",
        also: &[],
        released: "2025-10-29",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(442_119),
        status: Status::Replaying,
    },
    Group {
        slug: "fara1.5-27b",
        model: "microsoft/Fara1.5-27B",
        also: &[],
        released: "2026-07-17",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_543),
        status: Status::Replaying,
    },
    Group {
        slug: "fara1.5-9b",
        model: "microsoft/Fara1.5-9B",
        also: &["Fara1.5-4B"],
        released: "2026-05-12",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_543),
        status: Status::Replaying,
    },
    Group {
        slug: "functiongemma-270m-it",
        model: "google/functiongemma-270m-it",
        also: &[],
        released: "2025-10-08",
        smg: (Some("json"), None),
        table: Some(Table::FunctionGemma),
        set: Some(246_750),
        status: Status::Replaying,
    },
    Group {
        slug: "gemma-3-12b-it-qat-q4-0-unquantized",
        model: "google/gemma-3-12b-it-qat-q4_0-unquantized",
        also: &[
            "gemma-3-27b-it-qat-q4_0-unquantized",
            "gemma-3-4b-it-qat-q4_0-unquantized",
        ],
        released: "2025-04-08",
        smg: (Some("json"), None),
        table: None,
        set: Some(184_969),
        status: Status::Pending,
    },
    Group {
        slug: "gemma-3-1b-it",
        model: "google/gemma-3-1b-it",
        also: &[],
        released: "2025-03-10",
        smg: (Some("json"), None),
        table: None,
        set: Some(184_969),
        status: Status::Pending,
    },
    Group {
        slug: "gemma-3-4b-it",
        model: "google/gemma-3-4b-it",
        also: &["gemma-3-12b-it", "gemma-3-27b-it"],
        released: "2025-02-20",
        smg: (Some("json"), None),
        table: None,
        set: Some(184_969),
        status: Status::Pending,
    },
    Group {
        slug: "gemma-4-e4b-it",
        model: "google/gemma-4-E4B-it",
        also: &["gemma-4-E2B-it"],
        released: "2026-03-02",
        smg: (Some("json"), None),
        table: Some(Table::Gemma4),
        set: Some(478_241),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-4.5v",
        model: "zai-org/GLM-4.5V",
        also: &[],
        released: "2025-08-10",
        smg: (Some("glm45_moe"), None),
        table: Some(Table::Glm4),
        set: Some(492_389),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-4.6",
        model: "zai-org/GLM-4.6",
        also: &["GLM-4.5", "GLM-4.5-Air"],
        released: "2025-09-29",
        smg: (Some("glm45_moe"), None),
        table: Some(Table::Glm4),
        set: Some(492_389),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-4.6v",
        model: "zai-org/GLM-4.6V",
        also: &[],
        released: "2025-12-07",
        smg: (Some("glm45_moe"), None),
        table: Some(Table::Glm4),
        set: Some(492_389),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-4.6v-flash",
        model: "zai-org/GLM-4.6V-Flash",
        also: &[],
        released: "2025-12-07",
        smg: (Some("glm45_moe"), None),
        table: Some(Table::Glm4),
        set: Some(492_389),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-4.7",
        model: "zai-org/GLM-4.7",
        also: &[],
        released: "2025-12-22",
        smg: (Some("glm47_moe"), None),
        table: Some(Table::Glm),
        set: Some(1_947),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-4.7-flash",
        model: "zai-org/GLM-4.7-Flash",
        also: &[],
        released: "2026-01-19",
        smg: (Some("glm47_moe"), None),
        table: Some(Table::Glm),
        set: Some(1_947),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-5",
        model: "zai-org/GLM-5",
        also: &[],
        released: "2026-02-11",
        smg: (Some("glm47_moe"), Some("glm45")),
        table: Some(Table::Glm),
        set: Some(1_947),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-5.1",
        model: "zai-org/GLM-5.1",
        also: &[],
        released: "2026-04-03",
        smg: (Some("glm47_moe"), Some("glm45")),
        table: Some(Table::Glm),
        set: Some(163_746),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-5.2",
        model: "zai-org/GLM-5.2",
        also: &[],
        released: "2026-06-16",
        smg: (Some("glm47_moe"), Some("glm45")),
        table: Some(Table::Glm),
        set: Some(492_389),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-5.3",
        model: "zai-org/GLM-5.3",
        also: &["GLM-5.3-BF16"],
        released: "2026-08-25",
        smg: (Some("glm47_moe"), Some("glm45")),
        table: Some(Table::Glm),
        set: Some(503_273),
        status: Status::Replaying,
    },
    Group {
        slug: "glm-5.3-flash",
        model: "zai-org/GLM-5.3-Flash",
        also: &["GLM-5.3-Flash-BF16"],
        released: "2026-08-25",
        smg: (Some("glm47_moe"), Some("glm45")),
        table: Some(Table::Glm),
        set: Some(503_273),
        status: Status::Replaying,
    },
    Group {
        slug: "granite-4.1-3b",
        model: "ibm-granite/granite-4.1-3b",
        also: &["granite-4.1-30b"],
        released: "2026-04-06",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "granite-speech-3.2-8b",
        model: "ibm-granite/granite-speech-3.2-8b",
        also: &[],
        released: "2025-03-26",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "granite-vision-3.1-2b-preview",
        model: "ibm-granite/granite-vision-3.1-2b-preview",
        also: &[],
        released: "2025-01-27",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "granite-vision-3.2-2b",
        model: "ibm-granite/granite-vision-3.2-2b",
        also: &[],
        released: "2025-02-17",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "granite-vision-3.3-2b",
        model: "ibm-granite/granite-vision-3.3-2b",
        also: &[],
        released: "2025-06-03",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "h2o-lightning-4b",
        model: "h2oai/h2o-lightning-4b",
        also: &[],
        released: "2026-10-02",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_543),
        status: Status::Replaying,
    },
    Group {
        slug: "hermes-4-14b",
        model: "NousResearch/Hermes-4-14B",
        also: &[],
        released: "2025-08-30",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "huatuogpt-3-27b",
        model: "FreedomIntelligence/HuatuoGPT-3-27B",
        also: &[],
        released: "2026-09-24",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_563),
        status: Status::Replaying,
    },
    Group {
        slug: "hunyuan-a13b-instruct",
        model: "tencent/Hunyuan-A13B-Instruct",
        also: &[],
        released: "2025-06-25",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "hy4-preview",
        model: "tencent/Hy4-preview",
        also: &[],
        released: "2026-08-27",
        smg: (Some("hy_v4"), Some("hy_v4")),
        table: Some(Table::Hy4),
        set: Some(589_298),
        status: Status::Replaying,
    },
    Group {
        slug: "inkling",
        model: "thinkingmachines/Inkling",
        also: &["Inkling-Small"],
        released: "2026-07-14",
        smg: (Some("inkling"), Some("inkling")),
        table: Some(Table::Inkling),
        set: Some(589_659),
        status: Status::Replaying,
    },
    Group {
        slug: "iquest-q1",
        model: "IQuestLab/IQuest-Q1",
        also: &[],
        released: "2026-09-28",
        smg: (None, None),
        table: Some(Table::IQuest),
        set: Some(503_273),
        status: Status::Replaying,
    },
    Group {
        slug: "k-exaone-236b-a23b",
        model: "LGAI-EXAONE/K-EXAONE-236B-A23B",
        also: &[],
        released: "2025-12-26",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(503_634),
        status: Status::Replaying,
    },
    Group {
        slug: "k2-horizon-36b",
        model: "IFM/K2-Horizon-36B",
        also: &["K2-Horizon-32B", "K2-Horizon-375B-A23B"],
        released: "2026-09-01",
        smg: (None, None),
        table: Some(Table::K2Horizon),
        set: Some(12_814),
        status: Status::Replaying,
    },
    Group {
        slug: "kimi-k3",
        model: "moonshotai/Kimi-K3",
        also: &[],
        released: "2026-06-13",
        smg: (Some("kimi_k3"), Some("kimi_k3")),
        table: Some(Table::KimiK3),
        set: Some(54_418),
        status: Status::Ready,
    },
    Group {
        slug: "laguna-xs.2",
        model: "poolside/Laguna-XS.2",
        also: &[],
        released: "2026-04-23",
        smg: (None, None),
        table: Some(Table::Glm4),
        set: Some(490_452),
        status: Status::Replaying,
    },
    Group {
        slug: "lfm2.5-1.2b-instruct",
        model: "LiquidAI/LFM2.5-1.2B-Instruct",
        also: &["LFM2.5-1.2B-Thinking"],
        released: "2026-01-06",
        smg: (None, None),
        table: Some(Table::Lfm2_5),
        set: Some(589_657),
        status: Status::Replaying,
    },
    Group {
        slug: "ling-3.0-flash",
        model: "inclusionAI/Ling-3.0-flash",
        also: &["Ling-3.0-flash-fp4"],
        released: "2026-08-02",
        smg: (None, None),
        table: Some(Table::Ling),
        set: Some(589_298),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-3-1-nemotron-ultra-253b-cpt-v1",
        model: "nvidia/Llama-3_1-Nemotron-Ultra-253B-CPT-v1",
        also: &[],
        released: "2025-04-08",
        smg: (Some("json"), None),
        table: Some(Table::Json),
        set: Some(458_185),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-3-1-nemotron-ultra-253b-v1",
        model: "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
        also: &[],
        released: "2025-04-07",
        smg: (Some("json"), None),
        table: None,
        set: Some(328_643),
        status: Status::Pending,
    },
    Group {
        slug: "llama-3-3-nemotron-super-49b-v1-5",
        model: "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        also: &[],
        released: "2025-07-25",
        smg: (Some("json"), Some("nano_v3")),
        table: Some(Table::LlamaNemotron),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-3.1-nemotron-8b-ultralong-1m-instruct",
        model: "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct",
        also: &[],
        released: "2025-03-04",
        smg: (Some("json"), None),
        table: Some(Table::Json),
        set: Some(458_185),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-3.1-nemotron-8b-ultralong-2m-instruct",
        model: "nvidia/Llama-3.1-Nemotron-8B-UltraLong-2M-Instruct",
        also: &[],
        released: "2025-03-04",
        smg: (Some("json"), None),
        table: Some(Table::Json),
        set: Some(458_185),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-3.1-nemotron-8b-ultralong-4m-instruct",
        model: "nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct",
        also: &[],
        released: "2025-03-04",
        smg: (Some("json"), None),
        table: Some(Table::Json),
        set: Some(458_185),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-3.1-nemotron-nano-4b-v1.1",
        model: "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        also: &[],
        released: "2025-05-03",
        smg: (Some("json"), Some("nano_v3")),
        table: None,
        set: Some(576_772),
        status: Status::Pending,
    },
    Group {
        slug: "llama-3.1-nemotron-nano-8b-v1",
        model: "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        also: &[],
        released: "2025-03-16",
        smg: (Some("json"), Some("nano_v3")),
        table: Some(Table::LlamaNemotron),
        set: Some(416_850),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-3.1-nemotron-nano-vl-8b-v1-fp4-qad",
        model: "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD",
        also: &[],
        released: "2025-10-01",
        smg: (Some("json"), Some("nano_v3")),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "llama-3.1-nemotron-safety-guard-8b-v3",
        model: "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
        also: &[],
        released: "2025-08-20",
        smg: (Some("json"), None),
        table: Some(Table::Json),
        set: Some(458_185),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-3.3-nemotron-70b-reward-principle",
        model: "nvidia/Llama-3.3-Nemotron-70B-Reward-Principle",
        also: &[
            "Llama-3.3-Nemotron-70B-Edit",
            "Llama-3.3-Nemotron-70B-Feedback",
        ],
        released: "2025-10-12",
        smg: (Some("json"), None),
        table: Some(Table::Json),
        set: Some(458_185),
        status: Status::Replaying,
    },
    Group {
        slug: "llama-xlam-2-8b-fc-r",
        model: "Salesforce/Llama-xLAM-2-8b-fc-r",
        also: &[],
        released: "2025-03-27",
        smg: (Some("json"), None),
        table: Some(Table::Xlam),
        set: Some(483_062),
        status: Status::Replaying,
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
        also: &["MiMo-V2.5-Base"],
        released: "2026-04-27",
        smg: (None, None),
        table: Some(Table::Qwen3TaggedDirect),
        set: Some(589_278),
        status: Status::Replaying,
    },
    Group {
        slug: "minicpm5-2b",
        model: "openbmb/MiniCPM5-2B",
        also: &[],
        released: "2026-09-06",
        smg: (None, None),
        table: Some(Table::MiniCpm5),
        set: Some(589_277),
        status: Status::Replaying,
    },
    Group {
        slug: "minimax-m2",
        model: "MiniMaxAI/MiniMax-M2",
        also: &[],
        released: "2025-10-22",
        smg: (Some("minimax_m2"), Some("minimax")),
        table: Some(Table::MinimaxM2),
        set: Some(12_821),
        status: Status::Replaying,
    },
    Group {
        slug: "minimax-m2.5",
        model: "MiniMaxAI/MiniMax-M2.5",
        also: &[],
        released: "2026-02-12",
        smg: (Some("minimax_m2"), Some("minimax")),
        table: Some(Table::MinimaxM2),
        set: Some(12_821),
        status: Status::Replaying,
    },
    Group {
        slug: "minimax-m2.7",
        model: "MiniMaxAI/MiniMax-M2.7",
        also: &[],
        released: "2026-04-09",
        smg: (Some("minimax_m2"), Some("minimax")),
        table: Some(Table::MinimaxM2),
        set: Some(12_821),
        status: Status::Replaying,
    },
    Group {
        slug: "minimax-m3",
        model: "MiniMaxAI/MiniMax-M3",
        also: &[],
        released: "2026-06-02",
        smg: (Some("minimax_m3"), Some("minimax_m3")),
        table: Some(Table::MinimaxM3),
        set: Some(589_302),
        status: Status::Replaying,
    },
    Group {
        slug: "ministral-3-14b-instruct-2512-bf16",
        model: "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        also: &[],
        released: "2025-10-31",
        smg: (None, None),
        table: Some(Table::Mistral),
        set: Some(570_194),
        status: Status::Replaying,
    },
    Group {
        slug: "ministral-3-14b-reasoning-2512",
        model: "mistralai/Ministral-3-14B-Reasoning-2512",
        also: &[
            "Ministral-3-3B-Reasoning-2512",
            "Ministral-3-8B-Reasoning-2512",
        ],
        released: "2025-10-31",
        smg: (None, None),
        table: Some(Table::Mistral),
        set: Some(583_079),
        status: Status::Replaying,
    },
    Group {
        slug: "ministral-3-3b-instruct-2512",
        model: "mistralai/Ministral-3-3B-Instruct-2512",
        also: &[],
        released: "2025-10-31",
        smg: (None, None),
        table: Some(Table::Mistral),
        set: Some(570_194),
        status: Status::Replaying,
    },
    Group {
        slug: "ministral-3-3b-instruct-2512-bf16",
        model: "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        also: &[],
        released: "2025-10-31",
        smg: (None, None),
        table: Some(Table::Mistral),
        set: Some(570_194),
        status: Status::Replaying,
    },
    Group {
        slug: "ministral-3-8b-instruct-2512-bf16",
        model: "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        also: &[],
        released: "2025-10-31",
        smg: (None, None),
        table: Some(Table::Mistral),
        set: Some(570_194),
        status: Status::Replaying,
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
        slug: "mistral-medium-3.5-128b",
        model: "mistralai/Mistral-Medium-3.5-128B",
        also: &[],
        released: "2026-03-31",
        smg: (Some("mistral"), None),
        table: Some(Table::Mistral),
        set: Some(583_079),
        status: Status::Replaying,
    },
    Group {
        slug: "mistral-small-24b-instruct-2501",
        model: "mistralai/Mistral-Small-24B-Instruct-2501",
        also: &[],
        released: "2025-01-28",
        smg: (Some("mistral"), None),
        table: None,
        set: Some(184_972),
        status: Status::Pending,
    },
    Group {
        slug: "mistral-small-4-119b-2603",
        model: "mistralai/Mistral-Small-4-119B-2603",
        also: &[],
        released: "2026-01-23",
        smg: (Some("mistral"), None),
        table: Some(Table::Mistral),
        set: Some(583_079),
        status: Status::Replaying,
    },
    Group {
        slug: "muse-glimmer-30b",
        model: "meta-models/Muse-Glimmer-30B",
        also: &[],
        released: "2026-08-09",
        smg: (None, None),
        table: None,
        set: Some(494_673),
        status: Status::Pending,
    },
    Group {
        slug: "nanbeige4.2-3b",
        model: "Nanbeige/Nanbeige4.2-3B",
        also: &[],
        released: "2026-07-21",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(589_298),
        status: Status::Replaying,
    },
    Group {
        slug: "nemotron-cascade-14b-thinking",
        model: "nvidia/Nemotron-Cascade-14B-Thinking",
        also: &["Nemotron-Cascade-8B", "Nemotron-Cascade-8B-Thinking"],
        released: "2025-12-08",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "nemotron-cascade-2-30b-a3b",
        model: "nvidia/Nemotron-Cascade-2-30B-A3B",
        also: &[],
        released: "2026-03-18",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "nemotron-orchestrator-8b",
        model: "nvidia/Nemotron-Orchestrator-8B",
        also: &["IQuestLab/Fleming-R1-32B"],
        released: "2025-11-25",
        smg: (None, None),
        table: Some(Table::Qwen3),
        set: Some(583_086),
        status: Status::Replaying,
    },
    Group {
        slug: "nemotron-research-goosereason-4b-instruct",
        model: "nvidia/Nemotron-Research-GooseReason-4B-Instruct",
        also: &[
            "OpenMOSS-Team/SciJudge-4B",
            "OpenMOSS-Team/SciJudge-4B-2605",
        ],
        released: "2026-01-14",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "nvidia-nemotron-3-nano-30b-a3b-bf16",
        model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        also: &[],
        released: "2025-12-04",
        smg: (Some("qwen_xml"), Some("nano_v3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "nvidia-nemotron-3-nano-4b-bf16",
        model: "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        also: &[],
        released: "2026-03-07",
        smg: (Some("qwen_xml"), Some("nano_v3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "nvidia-nemotron-3-super-120b-a12b-bf16",
        model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        also: &[
            "Nemotron-3-Labs-Ultra-Math-SFT",
            "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        ],
        released: "2026-03-10",
        smg: (Some("qwen_xml"), Some("nano_v3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "nvidia-nemotron-3-ultra-550b-a55b-bf16",
        model: "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        also: &[
            "Nemotron-3-Labs-Ultra-Math-RL",
            "NVIDIA-Nemotron-Labs-Teacher-Chat",
            "NVIDIA-Nemotron-Labs-Teacher-Competition-Coding",
            "NVIDIA-Nemotron-Labs-Teacher-General-Reasoning",
            "NVIDIA-Nemotron-Labs-Teacher-Instruction-Following",
            "NVIDIA-Nemotron-Labs-Teacher-STEM",
        ],
        released: "2026-06-03",
        smg: (Some("qwen_xml"), Some("nano_v3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "nvidia-nemotron-3.5-lightning-30b-a3b-bf16",
        model: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        also: &[],
        released: "2026-08-01",
        smg: (Some("qwen_xml"), Some("nano_v3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "nvidia-nemotron-labs-3-elastic-30b-a3b-bf16",
        model: "nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16",
        also: &[],
        released: "2026-04-01",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16",
        model: "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16",
        also: &[],
        released: "2026-06-24",
        smg: (None, None),
        table: Some(Table::Qwen3Tagged),
        set: Some(12_829),
        status: Status::Replaying,
    },
    Group {
        slug: "nvidia-nemotronlabs-ai-for-media-sports-tennis",
        model: "nvidia/NVIDIA-NemotronLabs-AI-for-Media-Sports-Tennis",
        also: &[],
        released: "2026-09-01",
        smg: (None, None),
        table: None,
        set: Some(12_825),
        status: Status::Pending,
    },
    Group {
        slug: "olmo-3-7b-instruct",
        model: "allenai/Olmo-3-7B-Instruct",
        also: &[],
        released: "2025-11-19",
        smg: (None, None),
        table: Some(Table::Olmo3),
        set: Some(570_199),
        status: Status::Replaying,
    },
    Group {
        slug: "opencodereasoning-nemotron-7b",
        model: "nvidia/OpenCodeReasoning-Nemotron-7B",
        also: &[
            "OpenCodeReasoning-Nemotron-1.1-14B",
            "OpenCodeReasoning-Nemotron-1.1-32B",
            "OpenCodeReasoning-Nemotron-1.1-7B",
            "OpenCodeReasoning-Nemotron-14B",
            "OpenCodeReasoning-Nemotron-32B",
            "OpenCodeReasoning-Nemotron-32B-IOI",
        ],
        released: "2025-04-15",
        smg: (None, None),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "openmath-nemotron-14b",
        model: "nvidia/OpenMath-Nemotron-14B",
        also: &[
            "OpenMath-Nemotron-1.5B",
            "OpenMath-Nemotron-32B",
            "OpenMath-Nemotron-7B",
        ],
        released: "2025-04-22",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "openmath-nemotron-14b-kaggle",
        model: "nvidia/OpenMath-Nemotron-14B-Kaggle",
        also: &[],
        released: "2025-04-22",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "openreasoning-nemotron-1.5b",
        model: "nvidia/OpenReasoning-Nemotron-1.5B",
        also: &["OpenReasoning-Nemotron-32B", "OpenReasoning-Nemotron-7B"],
        released: "2025-07-15",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "openreasoning-nemotron-14b",
        model: "nvidia/OpenReasoning-Nemotron-14B",
        also: &[],
        released: "2025-07-15",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "phi-4-mini-instruct",
        model: "microsoft/Phi-4-mini-instruct",
        also: &[],
        released: "2025-02-19",
        smg: (None, None),
        table: Some(Table::Plain),
        set: Some(328_662),
        status: Status::Replaying,
    },
    Group {
        slug: "phi-4-multimodal-instruct",
        model: "microsoft/Phi-4-multimodal-instruct",
        also: &[],
        released: "2025-02-24",
        smg: (None, None),
        table: None,
        set: Some(328_662),
        status: Status::Pending,
    },
    Group {
        slug: "qwen-agentworld-35b-a3b",
        model: "Qwen/Qwen-AgentWorld-35B-A3B",
        also: &[],
        released: "2026-06-22",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_563),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen-drive-1.0-4b",
        model: "Qwen/Qwen-Drive-1.0-4B",
        also: &[],
        released: "2026-08-27",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_543),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen2.5-7b-instruct-1m",
        model: "Qwen/Qwen2.5-7B-Instruct-1M",
        also: &["Qwen2.5-14B-Instruct-1M"],
        released: "2025-01-23",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen2_5),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen2.5-omni-7b",
        model: "Qwen/Qwen2.5-Omni-7B",
        also: &["Qwen2.5-Omni-3B"],
        released: "2025-03-22",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen2_5),
        set: Some(328_662),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen2.5-vl-32b-instruct",
        model: "Qwen/Qwen2.5-VL-32B-Instruct",
        also: &[],
        released: "2025-03-21",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen2_5),
        set: Some(328_662),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen2.5-vl-7b-instruct",
        model: "Qwen/Qwen2.5-VL-7B-Instruct",
        also: &[
            "allenai/olmOCR-7B-0725",
            "allenai/olmOCR-7B-0825",
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-72B-Instruct",
        ],
        released: "2025-01-26",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen2_5),
        set: Some(328_662),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-30b-a3b",
        model: "Qwen/Qwen3-30B-A3B",
        also: &["Qwen3-235B-A22B"],
        released: "2025-04-27",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(583_086),
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
        status: Status::Ready,
    },
    Group {
        slug: "qwen3-30b-a3b-thinking-2507",
        model: "Qwen/Qwen3-30B-A3B-Thinking-2507",
        also: &["NousResearch/nomos-1", "Qwen3-235B-A22B-Thinking-2507"],
        released: "2025-07-29",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3-4b-instruct-2507",
        model: "Qwen/Qwen3-4B-Instruct-2507",
        also: &[],
        released: "2025-08-05",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(41_533),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3-4b-saferl",
        model: "Qwen/Qwen3-4B-SafeRL",
        also: &[],
        released: "2025-09-30",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3-4b-thinking-2507",
        model: "Qwen/Qwen3-4B-Thinking-2507",
        also: &[],
        released: "2025-08-05",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3-8b",
        model: "Qwen/Qwen3-8B",
        also: &[
            "NousResearch/NousCoder-14B",
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Qwen3-4B",
        ],
        released: "2025-04-27",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3-coder-30b-a3b-instruct",
        model: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        also: &["Qwen3-Coder-480B-A35B-Instruct"],
        released: "2025-07-31",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(490_451),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-coder-next",
        model: "Qwen/Qwen3-Coder-Next",
        also: &["Qwen3-Coder-Next-Base"],
        released: "2026-01-30",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(490_469),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-next-80b-a3b-instruct",
        model: "Qwen/Qwen3-Next-80B-A3B-Instruct",
        also: &[],
        released: "2025-09-09",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-next-80b-a3b-thinking",
        model: "Qwen/Qwen3-Next-80B-A3B-Thinking",
        also: &[],
        released: "2025-09-09",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(554_276),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-omni-30b-a3b-instruct",
        model: "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        also: &[],
        released: "2025-09-20",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3-omni-30b-a3b-thinking",
        model: "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        also: &[],
        released: "2025-09-15",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(54_418),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3-vl-235b-a22b-thinking",
        model: "Qwen/Qwen3-VL-235B-A22B-Thinking",
        also: &["Qwen3-VL-30B-A3B-Thinking"],
        released: "2025-09-22",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(554_276),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-vl-30b-a3b-instruct",
        model: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        also: &["Qwen3-VL-235B-A22B-Instruct"],
        released: "2025-09-30",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-vl-8b-instruct",
        model: "Qwen/Qwen3-VL-8B-Instruct",
        also: &[
            "BAAI/RoboBrain2.5-4B",
            "lightonai/OriOn-Qwen",
            "lightonai/OriOn-Qwen-SR1",
            "nvidia/Cosmos-Reason2-32B",
            "Qwen3-VL-2B-Instruct",
            "Qwen3-VL-32B-Instruct",
            "Qwen3-VL-4B-Instruct",
            "tencent/WeVisDoc-2B",
            "tencent/WeVisDoc-4B",
        ],
        released: "2025-10-11",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(576_772),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3-vl-8b-thinking",
        model: "Qwen/Qwen3-VL-8B-Thinking",
        also: &[
            "Qwen3-VL-2B-Thinking",
            "Qwen3-VL-32B-Thinking",
            "Qwen3-VL-4B-Thinking",
        ],
        released: "2025-10-11",
        smg: (Some("qwen"), Some("qwen3")),
        table: Some(Table::Qwen3),
        set: Some(554_276),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3.5-27b",
        model: "Qwen/Qwen3.5-27B",
        also: &[],
        released: "2026-02-24",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(54_179),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3.5-2b",
        model: "Qwen/Qwen3.5-2B",
        also: &["ATH-MaaS/OvisOCR2", "Qwen3.5-0.8B"],
        released: "2026-02-28",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(461_722),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3.5-35b-a3b",
        model: "Qwen/Qwen3.5-35B-A3B",
        also: &[
            "Qwen3.5-122B-A10B",
            "Qwen3.5-397B-A17B",
            "xlangai/Qwen-CUA",
            "xlangai/RoboFine-VLM-397B-A17B",
        ],
        released: "2026-02-24",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(474_543),
        status: Status::Replaying,
    },
    Group {
        slug: "qwen3.5-9b",
        model: "Qwen/Qwen3.5-9B",
        also: &["microsoft/FrogNano-4B-2609", "Qwen3.5-4B"],
        released: "2026-02-27",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(54_179),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3.6-27b",
        model: "Qwen/Qwen3.6-27B",
        also: &[],
        released: "2026-04-21",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(54_193),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3.6-35b-a3b",
        model: "Qwen/Qwen3.6-35B-A3B",
        also: &[],
        released: "2026-04-15",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(54_193),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3.8-2.4t-a95b",
        model: "Qwen/Qwen3.8-2.4T-A95B",
        also: &[],
        released: "2026-08-08",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(54_183),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3.8-27b",
        model: "Qwen/Qwen3.8-27B",
        also: &["JetBrains/Qwen3.8-3.6-27B-blend"],
        released: "2026-08-05",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(54_193),
        status: Status::Ready,
    },
    Group {
        slug: "qwen3.8-flash-next",
        model: "Qwen/Qwen3.8-Flash-Next",
        also: &[],
        released: "2026-08-24",
        smg: (Some("qwen_xml"), Some("qwen3")),
        table: Some(Table::Qwen3Tagged),
        set: Some(54_193),
        status: Status::Ready,
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
        also: &["Seed-OSS-36B-Base-woSyn"],
        released: "2025-08-20",
        smg: (None, None),
        table: Some(Table::SeedOss),
        set: Some(53_178),
        status: Status::Ready,
    },
    Group {
        slug: "step-3.5-flash",
        model: "stepfun-ai/Step-3.5-Flash",
        also: &[],
        released: "2026-02-01",
        smg: (Some("step3"), None),
        table: Some(Table::Qwen3Tagged),
        set: Some(54_181),
        status: Status::Ready,
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
        table: Some(Table::Plain),
        set: Some(27_865),
        status: Status::Ready,
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
        set: Some(54_418),
        status: Status::Ready,
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
        "{} groups, {} checkpoints, newest first.\n{} ready, {} replaying a recorded \
         set, {} on main awaiting fixtures, {} pending.",
        groups.len(),
        checkpoints,
        tally(|s| *s == Status::Ready),
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
            table = group.table.map_or_else(|| "—".to_string(), table_cell),
            set = group.set.map_or_else(|| "—".to_string(), count),
            status = group.status.cell(),
        );
    }
    out
}

fn readme_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("README.md")
}

/// The variable naming a directory to write every table's structural tag into, as a request
/// carries it, one file per table and request shape (`<Table>-required|optional-closed|open.json`),
/// the directory's earlier files of those names removed first; `tools/compile_structural_tags.py`
/// compiles them with xgrammar, the engines' grammar library.
const WRITE_TAGS_ENV: &str = "SYMPHONY_WRITE_TAGS";
/// How the written files end, one per request shape: what a run removes before it writes.
const TAG_FILE_ENDINGS: [&str; 4] = [
    "-required-closed.json",
    "-optional-closed.json",
    "-required-open.json",
    "-optional-open.json",
];

/// Tools whose schemas reach every value shape the grammars write: a required string, a number
/// with a bound, a string enum, a boolean, `null`, a list of strings, an object with properties, a
/// reference to a definition holding an enum, a property that pins no type, and a tool without
/// properties.
fn tag_tools() -> Vec<Tool> {
    let tool = |name: &str, parameters: serde_json::Value| Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    };
    vec![
        tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1},
                    "units": {"type": "string", "enum": ["c", "f"]},
                    "detailed": {"type": "boolean"},
                    "nothing": {"type": "null"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "options": {"type": "object", "properties": {
                        "limit": {"type": "number"},
                        "verbose": {"type": "boolean"},
                    }},
                    "scale": {"$ref": "#/$defs/Scale"},
                    "anything": {},
                },
                "required": ["city", "days"],
                "$defs": {"Scale": {"type": "string", "enum": ["small", "large"]}},
            }),
        ),
        tool("ping", json!({"type": "object"})),
    ]
}

#[test]
fn every_tables_structural_tag_is_written_when_asked() {
    let Ok(directory) = std::env::var(WRITE_TAGS_ENV) else {
        return;
    };
    let directory = PathBuf::from(directory);
    fs::create_dir_all(&directory).expect("the directory");
    // Only this run's tags: a payload an earlier run left for a table or a shape that no longer
    // derives one would be compiled as if it were current. Only files named as this test names
    // them go, so a directory holding anything else keeps it.
    for entry in fs::read_dir(&directory).expect("the directory's entries") {
        let path = entry.expect("an entry").path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if TAG_FILE_ENDINGS.iter().any(|ending| name.ends_with(ending)) {
            fs::remove_file(&path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
        }
    }
    let tools = tag_tools();
    let mut seen: Vec<String> = Vec::new();
    let mut written = 0;
    for table in GROUPS.iter().filter_map(|group| group.table) {
        let name = format!("{table:?}");
        if seen.contains(&name) {
            continue;
        }
        seen.push(name.clone());
        for (at_least_one, reasoning_open) in
            [(true, false), (false, false), (true, true), (false, true)]
        {
            let Some(grammar) = table.format().grammar(&tools, at_least_one, reasoning_open) else {
                continue;
            };
            let file = format!(
                "{name}-{}-{}.json",
                if at_least_one { "required" } else { "optional" },
                if reasoning_open { "open" } else { "closed" }
            );
            let payload =
                serde_json::to_string_pretty(&grammar.payload()).expect("the payload as JSON");
            fs::write(directory.join(&file), payload).unwrap_or_else(|why| panic!("{file}: {why}"));
            written += 1;
        }
    }
    assert!(written > 0, "no table wrote a tag");
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
fn the_registry_reads_every_group_with_a_table_and_no_other() {
    for group in GROUPS {
        assert_eq!(table_for(group.model), group.table, "{}", group.slug);
        let organisation = group.model.split('/').next().expect("an organisation");
        for also in group.also {
            // A repository name is one under the primary's organisation; a full id names its own.
            let id = if also.contains('/') {
                also.to_string()
            } else {
                format!("{organisation}/{also}")
            };
            assert_eq!(table_for(&id), group.table, "{}: {also}", group.slug);
        }
    }
    // And the other way: every group the registry knows is a group here, with the same
    // checkpoints and table, so neither list gains an entry the other lacks.
    let mut registered = 0;
    for (model, also, table) in groups() {
        let group = GROUPS
            .iter()
            .find(|group| group.model == model)
            .unwrap_or_else(|| panic!("{model}: in the registry, not in the README"));
        assert_eq!(group.also, also, "{model}");
        assert_eq!(group.table, Some(table), "{model}");
        registered += 1;
    }
    assert_eq!(
        registered,
        GROUPS.iter().filter(|group| group.table.is_some()).count()
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
