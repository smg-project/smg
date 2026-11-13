//! A retry derives a table's tag and a JSON answer's constraint for its own worker's engine and
//! grammar start.

use std::sync::Arc;

use serde_json::Value;
use smg_grpc_client::{sglang_proto, tokenspeed_proto, trtllm_proto, vllm_proto};

use super::*;
use crate::routers::grpc::{
    context::ExecutionPlanKind, proto_wrapper::ProtoRequest, testing::worker,
};

/// Each backend's built request carrying `tag` where its builder places a
/// forced call's structural tag.
fn requests_with_tag(tag: &str) -> Vec<(&'static str, ProtoGenerateRequest)> {
    vec![
        (
            "sglang",
            ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
                sampling_params: Some(sglang_proto::SamplingParams {
                    constraint: Some(sglang_proto::sampling_params::Constraint::StructuralTag(
                        tag.to_string(),
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        ),
        (
            "vllm",
            ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
                sampling_params: Some(vllm_proto::SamplingParams {
                    constraint: Some(vllm_proto::sampling_params::Constraint::StructuralTag(
                        tag.to_string(),
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        ),
        (
            "tokenspeed",
            ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
                sampling_params: Some(tokenspeed_proto::SamplingParams {
                    constraint: Some(
                        tokenspeed_proto::sampling_params::Constraint::StructuralTag(
                            tag.to_string(),
                        ),
                    ),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        ),
        (
            "trtllm",
            ProtoGenerateRequest::Trtllm(Box::new(trtllm_proto::GenerateRequest {
                guided_decoding: Some(trtllm_proto::GuidedDecodingParams {
                    guide_type: trtllm_proto::guided_decoding_params::GuideType::StructuralTag
                        as i32,
                    guide: tag.to_string(),
                }),
                ..Default::default()
            })),
        ),
    ]
}

fn plan_tag(plan: &ExecutionPlan) -> String {
    plan_request(plan)
        .structural_tag()
        .expect("the plan keeps a structural tag")
        .to_string()
}

/// The one generate request of a single plan.
fn plan_request(plan: &ExecutionPlan) -> &ProtoGenerateRequest {
    match plan {
        ExecutionPlan::Single(ProtoRequest::Generate(request)) => request,
        _ => panic!("expected a single generate plan"),
    }
}

/// A request that carries a JSON schema has no structural tag to write over
/// or clear: both leave the schema as built.
#[test]
fn a_request_carrying_a_schema_keeps_it_under_a_tags_write_and_clear() {
    let schema = r#"{"type":"object"}"#.to_string();
    let mut request = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
        sampling_params: Some(sglang_proto::SamplingParams {
            constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                schema.clone(),
            )),
            ..Default::default()
        }),
        ..Default::default()
    }));
    request.set_structural_tag("{}".to_string());
    assert_eq!(request.structural_tag(), None);
    request.clear_structural_tag();
    let ProtoGenerateRequest::Sglang(req) = &request else {
        panic!("the request was built for SGLang");
    };
    assert_eq!(
        req.sampling_params.as_ref().unwrap().constraint,
        Some(sglang_proto::sampling_params::Constraint::JsonSchema(
            schema
        ))
    );
}

/// A table's `auto` tag as a test derives it per start: the shape for an
/// engine starting after its reasoning parser, another for one starting
/// at the first token, and none for an engine that does not say.
#[derive(Debug)]
struct TableTag;

impl TagForStart for TableTag {
    fn tag_for(
        &self,
        start: GrammarStart,
        _vocabulary: utils::TagVocabulary,
    ) -> Result<Option<String>, String> {
        Ok(match start {
            GrammarStart::AfterReasoning => Some(BARE_TABLE_TAG.to_string()),
            GrammarStart::FirstToken => Some(THOUGHT_TABLE_TAG.to_string()),
            GrammarStart::Unknown => None,
        })
    }

    fn schema(&self) -> Option<&Value> {
        None
    }
}

/// A JSON answer's tag as a test derives it per start: the closed thought
/// then the schema for an engine starting at the first token, none for
/// the others, which take the request's own schema (held here) in its
/// place.
#[derive(Debug)]
struct AnswerTag(Value);

impl TagForStart for AnswerTag {
    fn tag_for(
        &self,
        start: GrammarStart,
        _vocabulary: utils::TagVocabulary,
    ) -> Result<Option<String>, String> {
        Ok((start == GrammarStart::FirstToken).then(|| THOUGHT_ANSWER_TAG.to_string()))
    }

    fn schema(&self) -> Option<&Value> {
        Some(&self.0)
    }
}

const ANSWER_SCHEMA: &str = r#"{"type":"object"}"#;
const THOUGHT_ANSWER_TAG: &str = r#"{"format":{"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text","excludes":[]},"end":"</think>"},{"type":"json_schema","json_schema":{"type":"object"}}]}}"#;

/// Each backend's built request carrying `schema` where its builder places
/// the request's own JSON schema.
fn requests_with_schema(schema: &str) -> Vec<(&'static str, ProtoGenerateRequest)> {
    vec![
        (
            "sglang",
            ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
                sampling_params: Some(sglang_proto::SamplingParams {
                    constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                        schema.to_string(),
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        ),
        (
            "vllm",
            ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
                sampling_params: Some(vllm_proto::SamplingParams {
                    constraint: Some(vllm_proto::sampling_params::Constraint::JsonSchema(
                        schema.to_string(),
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        ),
        (
            "tokenspeed",
            ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
                sampling_params: Some(tokenspeed_proto::SamplingParams {
                    constraint: Some(tokenspeed_proto::sampling_params::Constraint::JsonSchema(
                        schema.to_string(),
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        ),
        (
            "trtllm",
            ProtoGenerateRequest::Trtllm(Box::new(trtllm_proto::GenerateRequest {
                guided_decoding: Some(trtllm_proto::GuidedDecodingParams {
                    guide_type: trtllm_proto::guided_decoding_params::GuideType::JsonSchema as i32,
                    guide: schema.to_string(),
                }),
                ..Default::default()
            })),
        ),
    ]
}

/// The plan's constraint: its structural tag and its JSON schema, one of
/// them at most.
fn plan_constraint(plan: &ExecutionPlan) -> (Option<String>, Option<String>) {
    match plan {
        ExecutionPlan::Single(ProtoRequest::Generate(request)) => (
            request.structural_tag().map(str::to_string),
            request.json_schema().map(str::to_string),
        ),
        _ => panic!("expected a single generate plan"),
    }
}

/// A JSON answer's tag, shaped at build for the selected worker's start,
/// reaches a retry's worker derived again for that worker's own start, the
/// request's own schema standing in where the table gives it no tag: from
/// a first attempt that carried the tag and from one that carried the
/// schema alike, on every backend that carries a constraint. The `auto`
/// tag's derivation stands in for nothing, so a request built with the
/// schema captures none of it, as before.
#[test]
fn retry_derives_the_json_answers_tag_for_its_own_workers_start_or_its_schema() {
    let after_reasoning = worker(&[("constrained_decoding_start", "after_reasoning")]);
    let first_token = worker(&[("constrained_decoding_start", "first_token")]);
    let silent = worker(&[]);
    let answer: Arc<dyn TagForStart> =
        Arc::new(AnswerTag(serde_json::from_str(ANSWER_SCHEMA).unwrap()));
    let the_tag = (Some(THOUGHT_ANSWER_TAG.to_string()), None);
    let the_schema = (None, Some(ANSWER_SCHEMA.to_string()));

    let built = requests_with_tag(THOUGHT_ANSWER_TAG)
        .into_iter()
        .map(|(backend, request)| (backend, "the tag", request))
        .chain(
            requests_with_schema(ANSWER_SCHEMA)
                .into_iter()
                .map(|(backend, request)| (backend, "the schema", request)),
        );
    for (backend, carried, request) in built {
        let stamp = AttemptStamp {
            id: IdStamp::Exact,
            sampling_mask: None,
            sampling_baseline: None,
            inject_pd_metadata: false,
            tool_constraint: ToolConstraintStamp::capture_table(Arc::clone(&answer), &request),
        };
        assert!(
            stamp.tool_constraint.is_some(),
            "{backend}, built with {carried}: the answer's derivation goes with the plan"
        );
        let mut plan = ExecutionPlan::generate(ExecutionPlanKind::Single, request);

        restamp_plan_for_attempt(&mut plan, &stamp, &after_reasoning).unwrap();
        assert_eq!(
            plan_constraint(&plan),
            the_schema,
            "{backend}, built with {carried}: after the gate, the request's own schema"
        );

        restamp_plan_for_attempt(&mut plan, &stamp, &first_token).unwrap();
        assert_eq!(
            plan_constraint(&plan),
            the_tag,
            "{backend}, built with {carried}: from the first token, the closed thought then \
             the schema"
        );

        restamp_plan_for_attempt(&mut plan, &stamp, &silent).unwrap();
        assert_eq!(
            plan_constraint(&plan),
            the_schema,
            "{backend}, built with {carried}: a worker that does not say, the schema"
        );
    }

    for (backend, request) in requests_with_schema(ANSWER_SCHEMA) {
        let auto: Arc<dyn TagForStart> = Arc::new(TableTag);
        assert!(
            ToolConstraintStamp::capture_table(auto, &request).is_none(),
            "{backend}: the auto tag's derivation is not captured over the schema"
        );
    }
}

const BARE_TABLE_TAG: &str = r#"{"format":{"type":"const_string","value":"<call>"}}"#;
const THOUGHT_TABLE_TAG: &str = r#"{"format":{"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text","excludes":[]},"end":"</think>"},{"type":"const_string","value":"<call>"}]}}"#;

/// The table's `auto` tag, shaped at build for the selected worker's
/// start, reaches a retry's worker derived again for that worker's own
/// start: the other shape on an engine that starts the grammar elsewhere,
/// and no tag at all, the request left unconstrained, on a worker whose
/// engine the table gives none. On every backend that carries a
/// structural tag; a backend whose builder let the request's
/// `response_format` outrank the tag captures nothing.
#[test]
fn retry_derives_the_tables_auto_tag_for_its_own_workers_start() {
    let after_reasoning = worker(&[("constrained_decoding_start", "after_reasoning")]);
    let first_token = worker(&[("constrained_decoding_start", "first_token")]);
    let silent = worker(&[]);
    let tag: Arc<dyn TagForStart> = Arc::new(TableTag);

    for (backend, request) in requests_with_tag(BARE_TABLE_TAG) {
        let stamp = AttemptStamp {
            id: IdStamp::Exact,
            sampling_mask: None,
            sampling_baseline: None,
            inject_pd_metadata: false,
            tool_constraint: ToolConstraintStamp::capture_table(Arc::clone(&tag), &request),
        };
        assert!(
            stamp.tool_constraint.is_some(),
            "{backend}: the built request carries the table's tag"
        );
        let mut plan = ExecutionPlan::generate(ExecutionPlanKind::Single, request);

        restamp_plan_for_attempt(&mut plan, &stamp, &first_token).unwrap();
        assert_eq!(
            plan_tag(&plan),
            THOUGHT_TABLE_TAG,
            "{backend}: a retry on an engine that starts the grammar at the first token"
        );

        restamp_plan_for_attempt(&mut plan, &stamp, &after_reasoning).unwrap();
        assert_eq!(
            plan_tag(&plan),
            BARE_TABLE_TAG,
            "{backend}: a retry back on an engine that starts it after the reasoning"
        );

        restamp_plan_for_attempt(&mut plan, &stamp, &silent).unwrap();
        let ExecutionPlan::Single(ProtoRequest::Generate(request)) = &plan else {
            panic!("expected a single generate plan");
        };
        assert_eq!(
            request.structural_tag(),
            None,
            "{backend}: a worker the table gives no tag goes unconstrained"
        );

        // And back: the tag returns on the next worker the table shapes one for.
        restamp_plan_for_attempt(&mut plan, &stamp, &after_reasoning).unwrap();
        assert_eq!(
            plan_tag(&plan),
            BARE_TABLE_TAG,
            "{backend}: a retry after an unconstrained attempt takes the tag again"
        );
    }

    let schema = r#"{"type":"object"}"#.to_string();
    let outranked = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
        sampling_params: Some(sglang_proto::SamplingParams {
            constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                schema,
            )),
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert!(ToolConstraintStamp::capture_table(tag, &outranked).is_none());
}
