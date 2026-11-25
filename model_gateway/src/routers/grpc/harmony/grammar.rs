//! The vLLM whole-turn grammar for a Harmony tool call or JSON answer.

use openai_protocol::{common::Tool, worker::TagVocabulary};
use serde_json::{json, Value};
use smg_grpc_client::vllm_proto;

use super::super::{context::WorkerSelection, proto_wrapper::ProtoGenerateRequest};

const CHANNEL: &str = "<|channel|>";
const MESSAGE: &str = "<|message|>";
const START: &str = "<|start|>";
const END: &str = "<|end|>";
const CONSTRAIN: &str = "<|constrain|>";
const CONTROL: &[&str] = &[
    "<|return|>",
    "<|call|>",
    "<|endoftext|>",
    "<|startoftext|>",
    CHANNEL,
    MESSAGE,
    START,
    END,
    CONSTRAIN,
];
/// The alternate generation stops in o200k_harmony. Its tokenizer EOS
/// (<|return|>, 200002) remains available when XGrammar completes the result.
const ALTERNATE_STOP_IDS: [u32; 2] = [199999, 200012];

/// Keep one copy of each schema. An attempt renders it in the vocabulary
/// the selected workers support, including when a retry switches workers.
#[derive(Clone, Debug)]
pub(crate) struct VllmHarmonyGrammar {
    tools: Vec<(String, Value)>,
    answer: Option<Value>,
}

impl VllmHarmonyGrammar {
    pub(crate) fn new<'a>(
        tools: impl IntoIterator<Item = &'a Tool>,
        answer: Option<&Value>,
    ) -> Self {
        Self {
            tools: tools
                .into_iter()
                .map(|t| {
                    (
                        t.function.name.clone(),
                        t.function.parameters_schema().clone(),
                    )
                })
                .collect(),
            answer: answer.cloned(),
        }
    }

    pub(crate) fn apply(
        &self,
        request: &mut ProtoGenerateRequest,
        workers: Option<&WorkerSelection>,
    ) {
        let ProtoGenerateRequest::Vllm(request) = request else {
            return;
        };
        // Sequential PD submits the same structural tag to prefill and decode.
        let token_nodes = workers.is_some_and(|selection| {
            selection.workers().all(|worker| {
                worker
                    .metadata()
                    .spec
                    .labels
                    .get(TagVocabulary::LABEL)
                    .is_some_and(|v| v == "current")
            })
        });
        let params = request.sampling_params.get_or_insert_with(Default::default);
        params.constraint = Some(vllm_proto::sampling_params::Constraint::StructuralTag(
            self.render(token_nodes).to_string(),
        ));
        params.grammar_from_first_token = true;
        // The engine treats these IDs as stops, while its matcher normally
        // knows only the tokenizer EOS. A real <|call|> in a JSON string
        // would otherwise end generation with truncated arguments. RETURN
        // closes both final messages and addressed tool calls in Harmony.
        params.suppressed_token_ids = ALTERNATE_STOP_IDS.to_vec();
    }

    pub(crate) fn render(&self, token_nodes: bool) -> Value {
        let marker = |value: &str| {
            if token_nodes {
                json!({"type": "token", "token": value})
            } else {
                literal(value)
            }
        };
        let preamble = |name: &str| {
            let text = if token_nodes {
                json!({
                    "type": "tag", "begin": marker(MESSAGE), "end": marker(END),
                    "content": {"type": "any_tokens", "exclude_tokens": CONTROL},
                })
            } else {
                json!({
                    "type": "tag", "begin": MESSAGE, "end": END,
                    "content": {"type": "any_text", "excludes": ["<|call|>", "<|return|>", "<|endoftext|>"]},
                })
            };
            json!({
                "type": "or", "elements": [
                    {"type": "sequence", "elements": [marker(CHANNEL), literal(name), text, marker(START), literal("assistant")]},
                    literal(""),
                ]
            })
        };

        let mut outputs = Vec::new();
        for (name, schema) in &self.tools {
            let recipient = literal(&format!("commentary to=functions.{name}"));
            let content_type = json!({"type": "or", "elements": [
                {"type": "sequence", "elements": [marker(CONSTRAIN), literal("json")]},
                literal(""),
            ]});
            outputs.push(json!({
                "type": "sequence", "elements": [
                    marker(CHANNEL), recipient, content_type, marker(MESSAGE),
                    {"type": "json_schema", "json_schema": schema},
                ]
            }));
        }
        if let Some(schema) = &self.answer {
            outputs.push(json!({
                "type": "sequence", "elements": [
                    marker(CHANNEL), literal("final"), marker(MESSAGE),
                    {"type": "json_schema", "json_schema": schema},
                ]
            }));
        }
        json!({"format": {"type": "sequence", "elements": [
            preamble("analysis"), preamble("commentary"), {"type": "or", "elements": outputs},
        ]}})
    }
}

fn literal(text: &str) -> Value {
    json!({"type": "const_string", "value": text})
}
