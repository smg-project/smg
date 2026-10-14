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
//! other checkpoints under the primary's organisation; `tests/readme.rs` writes the README's
//! models table from the same groups and checks that the two agree.
//!
//! A lookup matches the repository name, the text after the last path separator (either slash),
//! exactly and without regard to case, under any organisation or path: `Qwen/Qwen3-8B`,
//! `qwen3-8b` and a local path ending in `Qwen3-8B` all read with the same table. Unlike the
//! gateway's parser registries, which match a pattern anywhere in the name, a quantized or
//! renamed checkpoint (`Qwen3-8B-FP8`) is not matched and gives `None`, so the gateway keeps its
//! older parsers for it until its group lists the name under `also`.

use super::{
    deepseek_v4_1, dots3, gemma4, glm, glm4, hy4, inkling, iquest, k2_horizon, kimi_k3, lfm2_5,
    ling, minicpm5, minimax_m2, minimax_m3, mistral, olmo3, plain, qwen2_5, qwen3, seed_oss, xlam,
};
use crate::{
    format::{CallSyntax, Format},
    tagged::{Placement, Spelling},
};

/// A table under [`formats`](super), by the function that builds it and, where the function
/// takes one, its call syntax. [`Table::name`] is how an operator names one: the gateway's
/// `--tool-call-parser` and `--reasoning-parser` flags, and a worker's labels, take a table's
/// name in place of a legacy parser's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    /// [`qwen3()`] with JSON calls.
    Qwen3,
    /// [`qwen3()`] with tagged calls, each value on its own line: Qwen 3.5 and later, Qwen3-Coder.
    Qwen3Tagged,
    /// [`qwen3()`] with tagged calls and the value between the tags directly: MiMo.
    Qwen3TaggedDirect,
    /// [`qwen2_5()`].
    Qwen2_5,
    /// [`deepseek_v4_1()`].
    DeepSeekV4_1,
    /// [`seed_oss()`].
    SeedOss,
    /// [`hy4()`].
    Hy4,
    /// [`ling()`].
    Ling,
    /// [`iquest()`].
    IQuest,
    /// [`olmo3()`].
    Olmo3,
    /// [`lfm2_5()`].
    Lfm2_5,
    /// [`xlam()`].
    Xlam,
    /// [`minimax_m2()`].
    MinimaxM2,
    /// [`minimax_m3()`].
    MinimaxM3,
    Mistral,
    /// [`kimi_k3()`].
    KimiK3,
    /// [`inkling()`].
    Inkling,
    /// [`minicpm5()`].
    MiniCpm5,
    /// [`dots3()`].
    Dots3,
    /// [`gemma4()`].
    Gemma4,
    /// [`glm()`].
    Glm,
    /// [`glm4()`].
    Glm4,
    /// [`k2_horizon()`].
    K2Horizon,
    /// [`plain()`].
    Plain,
}

impl Table {
    /// Every table, in this order.
    pub const ALL: [Table; 24] = [
        Self::Qwen3,
        Self::Qwen3Tagged,
        Self::Qwen3TaggedDirect,
        Self::Qwen2_5,
        Self::DeepSeekV4_1,
        Self::SeedOss,
        Self::Hy4,
        Self::Ling,
        Self::IQuest,
        Self::Olmo3,
        Self::Lfm2_5,
        Self::Xlam,
        Self::MinimaxM2,
        Self::MinimaxM3,
        Self::Mistral,
        Self::KimiK3,
        Self::Inkling,
        Self::MiniCpm5,
        Self::Dots3,
        Self::Gemma4,
        Self::Glm,
        Self::Glm4,
        Self::K2Horizon,
        Self::Plain,
    ];

    /// The table's name, as an operator spells it: its format's name, with the call syntax after
    /// it where the function takes one (`qwen3`, `qwen3_tagged`, `qwen3_tagged_direct`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Qwen3Tagged => "qwen3_tagged",
            Self::Qwen3TaggedDirect => "qwen3_tagged_direct",
            Self::Qwen2_5 => "qwen2.5",
            Self::DeepSeekV4_1 => "deepseek_v4.1",
            Self::SeedOss => "seed_oss",
            Self::Hy4 => "hy4",
            Self::Ling => "ling",
            Self::IQuest => "iquest",
            Self::Olmo3 => "olmo3",
            Self::Lfm2_5 => "lfm2.5",
            Self::Xlam => "xlam",
            Self::MinimaxM2 => "minimax_m2",
            Self::MinimaxM3 => "minimax_m3",
            Self::Mistral => "mistral",
            Self::KimiK3 => "kimi_k3",
            Self::Inkling => "inkling",
            Self::MiniCpm5 => "minicpm5",
            Self::Dots3 => "dots3",
            Self::Gemma4 => "gemma4",
            Self::Glm => "glm",
            Self::Glm4 => "glm4",
            Self::K2Horizon => "k2_horizon",
            Self::Plain => "plain",
        }
    }

    /// The table `name` spells ([`Table::name`], in any case, spaces around it ignored); `None`
    /// for any other text, a legacy parser's name among them.
    pub fn named(name: &str) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|table| table.name() == name)
    }

    /// The table, built.
    pub fn format(self) -> Format {
        match self {
            Self::Qwen3 => qwen3(CallSyntax::Json),
            Self::Qwen3Tagged => qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine)),
            Self::Qwen3TaggedDirect => qwen3(CallSyntax::Tagged(Spelling::Json, Placement::Direct)),
            Self::Qwen2_5 => qwen2_5(),
            Self::DeepSeekV4_1 => deepseek_v4_1(),
            Self::SeedOss => seed_oss(),
            Self::Hy4 => hy4(),
            Self::Ling => ling(),
            Self::IQuest => iquest(),
            Self::Olmo3 => olmo3(),
            Self::Lfm2_5 => lfm2_5(),
            Self::Xlam => xlam(),
            Self::MinimaxM2 => minimax_m2(),
            Self::MinimaxM3 => minimax_m3(),
            Self::Mistral => mistral(),
            Self::KimiK3 => kimi_k3(),
            Self::Inkling => inkling(),
            Self::MiniCpm5 => minicpm5(),
            Self::Dots3 => dots3(),
            Self::Gemma4 => gemma4(),
            Self::Glm => glm(),
            Self::Glm4 => glm4(),
            Self::K2Horizon => k2_horizon(),
            Self::Plain => plain(),
        }
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

    /// Whether `repository` names one of the group's checkpoints, in any case.
    fn holds(&self, repository: &str) -> bool {
        self.repositories()
            .any(|own| own.eq_ignore_ascii_case(repository))
    }
}

/// The repository name in `model`: the text after its last path separator, either slash, a
/// trailing one ignored.
fn repository(model: &str) -> &str {
    let model = model.trim_end_matches(['/', '\\']);
    model.rsplit(['/', '\\']).next().unwrap_or(model)
}

/// Every group whose table is on main, in bellwether's slug order.
const GROUPS: &[Group] = &[
    Group {
        model: "openbmb/AgentCPM-Explore",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "ai21labs/AI21-Jamba-Reasoning-3B",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "ai21labs/AI21-Jamba2-3B",
        also: &["AI21-Jamba2-Mini"],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "BAAI/AREX-2",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "BAAI/AREX-Base",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "BAAI/AREX-Turbo",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "allenai/AstaBrief_8B",
        also: &["AstaBrief_8B_SFT"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "internlm/Atria-Dawn-Preview",
        also: &["Atria-Dawn-Preview-Ascend-w8a8"],
        table: Table::Glm,
        opener: None,
    },
    Group {
        model: "baichuan-inc/Baichuan-M3-235B",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "internlm/CapRL-Qwen3VL-4B",
        also: &["CapRL-Qwen3VL-2B"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "internlm/CapRL-Video-4B",
        also: &["inclusionAI/ArmorOCR"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
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
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "nvidia/Cosmos3-Edge",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/Cosmos3-Edge-Policy-DROID",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "LiquidAI/d1-3B",
        also: &[],
        table: Table::Lfm2_5,
        opener: None,
    },
    Group {
        model: "deepseek-ai/DeepSeek-V4.1-Flash",
        also: &[],
        table: Table::DeepSeekV4_1,
        opener: None,
    },
    Group {
        model: "mistralai/Devstral-2-123B-Instruct-2512",
        also: &[],
        table: Table::Mistral,
        opener: None,
    },
    Group {
        model: "dots-studio/dots3-note-prev",
        also: &[],
        table: Table::Dots3,
        opener: None,
    },
    Group {
        model: "tiiuae/Falcon-H1-Tiny-90M-Instruct",
        also: &["Falcon-H1-Tiny-90M-Instruct-Curriculum"],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "tiiuae/Falcon-H1R-7B",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "google/gemma-4-E4B-it",
        also: &["gemma-4-E2B-it"],
        table: Table::Gemma4,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-4.5V",
        also: &[],
        table: Table::Glm4,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-4.6",
        also: &["GLM-4.5", "GLM-4.5-Air"],
        table: Table::Glm4,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-4.6V",
        also: &[],
        table: Table::Glm4,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-4.6V-Flash",
        also: &[],
        table: Table::Glm4,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-4.7",
        also: &[],
        table: Table::Glm,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-4.7-Flash",
        also: &[],
        table: Table::Glm,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-5",
        also: &[],
        table: Table::Glm,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-5.1",
        also: &[],
        table: Table::Glm,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-5.2",
        also: &[],
        table: Table::Glm,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-5.3",
        also: &["GLM-5.3-BF16"],
        table: Table::Glm,
        opener: None,
    },
    Group {
        model: "zai-org/GLM-5.3-Flash",
        also: &["GLM-5.3-Flash-BF16"],
        table: Table::Glm,
        opener: None,
    },
    Group {
        model: "ibm-granite/granite-4.1-3b",
        also: &["granite-4.1-30b"],
        table: Table::Qwen2_5,
        opener: Some("<|start_of_role|>assistant<|end_of_role|>"),
    },
    Group {
        model: "NousResearch/Hermes-4-14B",
        also: &[],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "tencent/Hy4-preview",
        also: &[],
        table: Table::Hy4,
        opener: None,
    },
    Group {
        model: "thinkingmachines/Inkling",
        also: &["Inkling-Small"],
        table: Table::Inkling,
        opener: None,
    },
    Group {
        model: "IQuestLab/IQuest-Q1",
        also: &[],
        table: Table::IQuest,
        opener: None,
    },
    Group {
        model: "LGAI-EXAONE/K-EXAONE-236B-A23B",
        also: &[],
        table: Table::Qwen3,
        opener: Some("<|assistant|>"),
    },
    Group {
        model: "IFM/K2-Horizon-36B",
        also: &["K2-Horizon-32B", "K2-Horizon-375B-A23B"],
        table: Table::K2Horizon,
        opener: None,
    },
    Group {
        model: "moonshotai/Kimi-K3",
        also: &[],
        table: Table::KimiK3,
        opener: None,
    },
    Group {
        model: "poolside/Laguna-XS.2",
        also: &[],
        table: Table::Glm4,
        opener: Some("<assistant>"),
    },
    Group {
        model: "LiquidAI/LFM2.5-1.2B-Instruct",
        also: &["LFM2.5-1.2B-Thinking"],
        table: Table::Lfm2_5,
        opener: None,
    },
    Group {
        model: "inclusionAI/Ling-3.0-flash",
        also: &["Ling-3.0-flash-fp4"],
        table: Table::Ling,
        opener: None,
    },
    Group {
        model: "Salesforce/Llama-xLAM-2-8b-fc-r",
        also: &[],
        table: Table::Xlam,
        opener: None,
    },
    Group {
        model: "XiaomiMiMo/MiMo-V2.5",
        also: &["MiMo-V2.5-Base"],
        table: Table::Qwen3TaggedDirect,
        opener: None,
    },
    Group {
        model: "openbmb/MiniCPM5-2B",
        also: &[],
        table: Table::MiniCpm5,
        opener: None,
    },
    Group {
        model: "MiniMaxAI/MiniMax-M2",
        also: &[],
        table: Table::MinimaxM2,
        opener: None,
    },
    Group {
        model: "MiniMaxAI/MiniMax-M2.5",
        also: &[],
        table: Table::MinimaxM2,
        opener: None,
    },
    Group {
        model: "MiniMaxAI/MiniMax-M2.7",
        also: &[],
        table: Table::MinimaxM2,
        opener: None,
    },
    Group {
        model: "MiniMaxAI/MiniMax-M3",
        also: &[],
        table: Table::MinimaxM3,
        opener: None,
    },
    Group {
        model: "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        also: &[],
        table: Table::Mistral,
        opener: None,
    },
    Group {
        model: "mistralai/Ministral-3-14B-Reasoning-2512",
        also: &[
            "Ministral-3-3B-Reasoning-2512",
            "Ministral-3-8B-Reasoning-2512",
        ],
        table: Table::Mistral,
        opener: None,
    },
    Group {
        model: "mistralai/Ministral-3-3B-Instruct-2512",
        also: &[],
        table: Table::Mistral,
        opener: None,
    },
    Group {
        model: "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        also: &[],
        table: Table::Mistral,
        opener: None,
    },
    Group {
        model: "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        also: &[],
        table: Table::Mistral,
        opener: None,
    },
    Group {
        model: "mistralai/Mistral-Medium-3.5-128B",
        also: &[],
        table: Table::Mistral,
        opener: None,
    },
    Group {
        model: "mistralai/Mistral-Small-4-119B-2603",
        also: &[],
        table: Table::Mistral,
        opener: None,
    },
    Group {
        model: "Nanbeige/Nanbeige4.2-3B",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/Nemotron-Cascade-14B-Thinking",
        also: &["Nemotron-Cascade-8B", "Nemotron-Cascade-8B-Thinking"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "nvidia/Nemotron-Cascade-2-30B-A3B",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/Nemotron-Orchestrator-8B",
        also: &["IQuestLab/Fleming-R1-32B"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "nvidia/Nemotron-Research-GooseReason-4B-Instruct",
        also: &[
            "OpenMOSS-Team/SciJudge-4B",
            "OpenMOSS-Team/SciJudge-4B-2605",
        ],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        also: &[
            "Nemotron-3-Labs-Ultra-Math-SFT",
            "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        ],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        also: &[
            "Nemotron-3-Labs-Ultra-Math-RL",
            "NVIDIA-Nemotron-Labs-Teacher-Chat",
            "NVIDIA-Nemotron-Labs-Teacher-Competition-Coding",
            "NVIDIA-Nemotron-Labs-Teacher-General-Reasoning",
            "NVIDIA-Nemotron-Labs-Teacher-Instruction-Following",
            "NVIDIA-Nemotron-Labs-Teacher-STEM",
        ],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "allenai/Olmo-3-7B-Instruct",
        also: &[],
        table: Table::Olmo3,
        opener: None,
    },
    Group {
        model: "nvidia/OpenCodeReasoning-Nemotron-7B",
        also: &[
            "OpenCodeReasoning-Nemotron-1.1-14B",
            "OpenCodeReasoning-Nemotron-1.1-32B",
            "OpenCodeReasoning-Nemotron-1.1-7B",
            "OpenCodeReasoning-Nemotron-14B",
            "OpenCodeReasoning-Nemotron-32B",
            "OpenCodeReasoning-Nemotron-32B-IOI",
        ],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "microsoft/Phi-4-mini-instruct",
        also: &[],
        table: Table::Plain,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen-AgentWorld-35B-A3B",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen-Drive-1.0-4B",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen2.5-7B-Instruct-1M",
        also: &["Qwen2.5-14B-Instruct-1M"],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen2.5-Omni-7B",
        also: &["Qwen2.5-Omni-3B"],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen2.5-VL-32B-Instruct",
        also: &[],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen2.5-VL-7B-Instruct",
        also: &[
            "allenai/olmOCR-7B-0725",
            "allenai/olmOCR-7B-0825",
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-72B-Instruct",
        ],
        table: Table::Qwen2_5,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-30B-A3B",
        also: &["Qwen3-235B-A22B"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-30B-A3B-Instruct-2507",
        also: &["Qwen3-235B-A22B-Instruct-2507"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-30B-A3B-Thinking-2507",
        also: &["NousResearch/nomos-1", "Qwen3-235B-A22B-Thinking-2507"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-4B-Instruct-2507",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-4B-SafeRL",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-4B-Thinking-2507",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-8B",
        also: &[
            "NousResearch/NousCoder-14B",
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Qwen3-4B",
        ],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        also: &["Qwen3-Coder-480B-A35B-Instruct"],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-Coder-Next",
        also: &["Qwen3-Coder-Next-Base"],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-Next-80B-A3B-Instruct",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-Next-80B-A3B-Thinking",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        also: &[],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-VL-235B-A22B-Thinking",
        also: &["Qwen3-VL-30B-A3B-Thinking"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        also: &["Qwen3-VL-235B-A22B-Instruct"],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
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
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3-VL-8B-Thinking",
        also: &[
            "Qwen3-VL-2B-Thinking",
            "Qwen3-VL-32B-Thinking",
            "Qwen3-VL-4B-Thinking",
        ],
        table: Table::Qwen3,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.5-27B",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.5-2B",
        also: &["ATH-MaaS/OvisOCR2", "Qwen3.5-0.8B"],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.5-35B-A3B",
        also: &[
            "Qwen3.5-122B-A10B",
            "Qwen3.5-397B-A17B",
            "xlangai/Qwen-CUA",
            "xlangai/RoboFine-VLM-397B-A17B",
        ],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.5-9B",
        also: &["microsoft/FrogNano-4B-2609", "Qwen3.5-4B"],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.6-27B",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.6-35B-A3B",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.8-2.4T-A95B",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.8-27B",
        also: &["JetBrains/Qwen3.8-3.6-27B-blend"],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "Qwen/Qwen3.8-Flash-Next",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "ByteDance-Seed/Seed-OSS-36B-Instruct",
        also: &["Seed-OSS-36B-Base-woSyn"],
        table: Table::SeedOss,
        opener: None,
    },
    Group {
        model: "stepfun-ai/Step-3.5-Flash",
        also: &[],
        table: Table::Qwen3Tagged,
        opener: None,
    },
    Group {
        model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        also: &[],
        table: Table::Plain,
        opener: None,
    },
    Group {
        model: "Qwen/WebWorld-32B",
        also: &["WebWorld-8B", "WebWorld-14B"],
        table: Table::Qwen3,
        opener: None,
    },
];

/// The group that holds `model`, by its repository name.
fn group_for(model: &str) -> Option<&'static Group> {
    let repository = repository(model);
    if repository.is_empty() {
        return None;
    }
    GROUPS.iter().find(|group| group.holds(repository))
}

/// The table that reads `model`'s output, with the model's own turn opener where its template has
/// one; `None` for a model the registry does not know.
pub fn for_model(model: &str) -> Option<Format> {
    let group = group_for(model)?;
    let format = group.table.format();
    Some(match group.opener {
        Some(opener) => format.opens_turn(opener),
        None => format,
    })
}

/// The table [`for_model`] builds for `model`, by name; `None` for a model the registry does not
/// know.
pub fn table_for(model: &str) -> Option<Table> {
    group_for(model).map(|group| group.table)
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
}
