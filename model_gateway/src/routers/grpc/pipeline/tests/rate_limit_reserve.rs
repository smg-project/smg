//! The rate-limit reserve stage and the settlement of its reservation, on every request shape the
//! pipeline serves.

use bytes::Bytes;
use llm_multimodal::registry::transcription::FAMILIES;
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    decisions::{DecisionInputTokensDetails, DecisionOutputTokensDetails},
    generate::{GenerateFinishReason, GenerateFinishType, GenerateMetaInfo, GenerateRequest},
    transcription::{AudioFile, TranscriptionRequest},
};
use serde_json::json;

use super::*;
use crate::{
    config::types::RouterConfig,
    rate_limit::{RateLimitManager, Reservation, ReserveRequest},
    routers::grpc::{
        context::EmbeddingInput, regular::symphony::template::Templates,
        spec::TranscriptionResponseFormat, ProcessedMessages,
    },
    tenant::TenantKey,
};

const MODEL: &str = "reserve-test-model";

async fn test_components() -> Arc<SharedComponents> {
    let worker_registry = Arc::new(WorkerRegistry::new());
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(tokenizer_registry.clone())),
        tokenizer_registry,
        worker_registry,
        tool_parser_factory: ToolParserFactory::default(),
        reasoning_parser_factory: ReasoningParserFactory::default(),
        parser_resolver: utils::ParserResolver::disabled(),
        multimodal: None,
        tokenizer_autoload: true,
        tokenizer_load_failures: Arc::default(),
        parser: ParserChoice::Legacy,
    })
}

/// Real `RateLimitManager` backed by a tiny in-memory budget --
/// `RateLimitManager::new` is private to the `rate_limit` module, so
/// `from_config` (with a temp YAML) is the only way to build one from
/// here. The YAML is read synchronously inside `from_config`, before the
/// tempdir is dropped at the end of this function, so no leak is needed.
fn manager_with_tokens_per_minute(tpm: u32) -> Arc<RateLimitManager> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rate_limit.yaml");
    std::fs::write(
        &path,
        format!("default_policy:\n  tokens_per_minute: {tpm}\n  requests_per_minute: 1000\n"),
    )
    .unwrap();
    let rc = RouterConfig::builder()
        .worker_startup_timeout_secs(1)
        .tenant_rate_limit_enabled(true)
        .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()))
        .build_unchecked();
    RateLimitManager::from_config(&rc).unwrap().unwrap()
}

fn ctx_with_tokens(components: Arc<SharedComponents>, num_tokens: usize) -> RequestContext {
    let request: GenerateRequest = serde_json::from_value(json!({
        "model": MODEL,
        "text": "Hello"
    }))
    .unwrap();
    let mut ctx =
        RequestContext::for_generate(Arc::new(request), None, MODEL.to_string(), components);
    ctx.state.preparation = Some(PreparationOutput::Generate {
        original_text: None,
        token_ids: vec![0; num_tokens],
    });
    ctx.input.tenant_request_meta = Some(TenantRequestMeta::new(TenantKey::new("test-tenant")));
    ctx
}

#[tokio::test]
async fn reserve_is_admitted_once_and_cached_on_the_cell() {
    // Admits one 5-token reservation but not two (5 + 5 > 6).
    let manager = manager_with_tokens_per_minute(6);
    let components = test_components().await;
    let mut ctx = ctx_with_tokens(components, 5);
    ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

    let stage = RateLimitReserveStage::new(Some(manager));

    assert!(stage.execute(&mut ctx).await.is_ok());
    // A second pass (defensive; ingress runs once) sees the cached
    // Admitted outcome and does not reserve again.
    assert!(stage.execute(&mut ctx).await.is_ok());

    assert!(matches!(
        ctx.input.rate_limit_cell.as_ref().unwrap().peek(),
        Some(RateLimitOutcome::Admitted(_))
    ));
}

#[tokio::test]
async fn denial_is_cached_on_the_cell() {
    // Too small for even one 5-token reservation.
    let manager = manager_with_tokens_per_minute(3);
    let components = test_components().await;
    let mut ctx = ctx_with_tokens(components, 5);
    ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

    let stage = RateLimitReserveStage::new(Some(manager));

    let response = stage.execute(&mut ctx).await.expect_err("should be denied");
    assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);

    assert!(matches!(
        ctx.input.rate_limit_cell.as_ref().unwrap().peek(),
        Some(RateLimitOutcome::Denied)
    ));
}

#[tokio::test]
async fn disabled_manager_is_a_no_op() {
    let components = test_components().await;
    let mut ctx = ctx_with_tokens(components, 5);
    ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

    let stage = RateLimitReserveStage::new(None);
    assert!(stage.execute(&mut ctx).await.is_ok());
    assert!(ctx.input.rate_limit_cell.as_ref().unwrap().peek().is_none());
}

#[tokio::test]
async fn endpoint_without_a_cell_is_a_no_op() {
    // The Responses iterations set no rate_limit_cell.
    let manager = manager_with_tokens_per_minute(6);
    let components = test_components().await;
    let mut ctx = ctx_with_tokens(components, 5);

    let stage = RateLimitReserveStage::new(Some(manager));
    assert!(stage.execute(&mut ctx).await.is_ok());
}

#[tokio::test]
async fn settle_reservation_with_zero_usage_refunds_the_full_estimate() {
    // Budget 6; reserve 5 leaves 1. A response without usage settles
    // with all zeros (see `recorded_settlement`, `completion_settlement`
    // and `generate_settlement`) -- that must refund the full 5-token
    // debit back to the tenant, not silently keep it (nothing was
    // actually consumed) and not leave the reservation stuck open
    // (which would leak budget forever).
    let manager = manager_with_tokens_per_minute(6);
    let components = test_components().await;
    let mut ctx = ctx_with_tokens(components, 5);
    ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

    let stage = RateLimitReserveStage::new(Some(manager.clone()));
    assert!(stage.execute(&mut ctx).await.is_ok());

    RequestPipeline::settle_reservation(
        ctx.input.rate_limit_cell.as_deref(),
        UsageSettlement::default(),
    )
    .await;

    // The full 6-token budget must be available again -- would be
    // denied if settle had kept the original 5-token debit instead of
    // truing it up to the real (zero) usage.
    let check = ReserveRequest {
        request_charge_id: uuid::Uuid::now_v7(),
        tenant_key: TenantKey::new("test-tenant"),
        model_id: Some(MODEL),
        estimated_input_tokens: 6,
    };
    assert!(matches!(
        manager.reserve(check).await,
        Reservation::Admitted(_)
    ));
}

/// The cached and reasoning split of a settlement is carried for the
/// consumers that price or report it; the true-up charges the counts.
#[tokio::test]
async fn settle_reservation_charges_the_counts_and_not_their_split() {
    let manager = manager_with_tokens_per_minute(100);
    let components = test_components().await;
    let mut ctx = ctx_with_tokens(components, 5);
    ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

    let stage = RateLimitReserveStage::new(Some(manager.clone()));
    assert!(stage.execute(&mut ctx).await.is_ok());

    RequestPipeline::settle_reservation(
        ctx.input.rate_limit_cell.as_deref(),
        UsageSettlement {
            actual_input_tokens: 5,
            completion_tokens: 10,
            cached_input_tokens: 4,
            reasoning_tokens: 3,
        },
    )
    .await;

    // 100 - 15 = 85 remain: the split neither charges twice nor refunds.
    let check = |tokens: u32| ReserveRequest {
        request_charge_id: uuid::Uuid::now_v7(),
        tenant_key: TenantKey::new("test-tenant"),
        model_id: Some(MODEL),
        estimated_input_tokens: tokens,
    };
    assert!(matches!(
        manager.reserve(check(85)).await,
        Reservation::Admitted(_)
    ));
    assert!(matches!(
        manager.reserve(check(1)).await,
        Reservation::Denied { .. }
    ));
}

/// The non-streamed completion settles from its usage object: the
/// engine's counters, their details included; zeros without one.
#[test]
fn completion_settlement_carries_the_usage_details() {
    let usage = Usage::from_counts(7, 5)
        .with_cached_tokens(3)
        .with_reasoning_tokens(2);
    assert_eq!(
        RequestPipeline::completion_settlement(Some(&usage)),
        UsageSettlement {
            actual_input_tokens: 7,
            completion_tokens: 5,
            cached_input_tokens: 3,
            reasoning_tokens: 2,
        }
    );
    assert_eq!(
        RequestPipeline::completion_settlement(Some(&Usage::from_counts(7, 5))),
        UsageSettlement {
            actual_input_tokens: 7,
            completion_tokens: 5,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        }
    );
    assert_eq!(
        RequestPipeline::completion_settlement(None),
        UsageSettlement::default()
    );
}

/// The non-streamed generate answer settles its shared prompt and cache
/// hit once, the greatest any choice reports, and sums each choice's
/// completion and reasoning counts; an empty answer settles zeros.
#[test]
fn generate_settlement_takes_the_shared_prompt_once_and_sums_the_choices() {
    let choice = |completion_tokens: u32, reasoning_tokens: Option<u32>, cached_tokens: u32| {
        GenerateResponse {
            text: String::new(),
            output_ids: Vec::new(),
            meta_info: GenerateMetaInfo {
                id: "gen".to_string(),
                finish_reason: GenerateFinishReason::Stop {
                    finish_type: GenerateFinishType::Stop,
                },
                prompt_tokens: 9,
                weight_version: "default".to_string(),
                input_token_logprobs: None,
                output_token_logprobs: None,
                completion_tokens,
                cached_tokens,
                reasoning_tokens,
                e2e_latency: 0.0,
                matched_stop: None,
            },
        }
    };
    assert_eq!(
        RequestPipeline::generate_settlement(&[choice(3, Some(1), 4), choice(5, None, 4)]),
        UsageSettlement {
            actual_input_tokens: 9,
            completion_tokens: 8,
            cached_input_tokens: 4,
            reasoning_tokens: 1,
        }
    );
    // The choices report different cache hits: the settlement takes the
    // greatest, as the streamed answer settles.
    assert_eq!(
        RequestPipeline::generate_settlement(&[choice(3, Some(1), 2), choice(5, None, 6)])
            .cached_input_tokens,
        6
    );
    assert_eq!(
        RequestPipeline::generate_settlement(&[]),
        UsageSettlement::default()
    );
}

/// A decision batch settles its input tokens and the cache and reasoning
/// counters its usage carries, no completion; a count past `u32` saturates.
#[test]
fn decisions_settlement_carries_the_usage_counters() {
    let usage = DecisionUsage {
        input_tokens: 11,
        output_tokens: 0,
        total_tokens: 11,
        input_tokens_details: DecisionInputTokensDetails {
            cached_tokens: 2,
            cache_write_tokens: 0,
        },
        output_tokens_details: DecisionOutputTokensDetails {
            reasoning_tokens: 1,
        },
    };
    assert_eq!(
        RequestPipeline::decisions_settlement(&usage),
        UsageSettlement {
            actual_input_tokens: 11,
            completion_tokens: 0,
            cached_input_tokens: 2,
            reasoning_tokens: 1,
        }
    );
    let beyond = DecisionUsage {
        input_tokens: u64::from(u32::MAX) + 1,
        ..usage
    };
    assert_eq!(
        RequestPipeline::decisions_settlement(&beyond).actual_input_tokens,
        u32::MAX
    );
}

fn with_tenant_and_cell(mut ctx: RequestContext) -> RequestContext {
    ctx.input.tenant_request_meta = Some(TenantRequestMeta::new(TenantKey::new("test-tenant")));
    ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));
    ctx
}

/// An embeddings request after its preparation: `num_tokens` input tokens.
fn embedding_ctx(components: Arc<SharedComponents>, num_tokens: usize) -> RequestContext {
    let request: EmbeddingRequest =
        serde_json::from_value(json!({"model": MODEL, "input": "Hello"})).unwrap();
    let mut ctx =
        RequestContext::for_embedding(Arc::new(request), None, MODEL.to_string(), components);
    ctx.state.preparation = Some(PreparationOutput::Embedding {
        inputs: vec![EmbeddingInput {
            text: Some("Hello".to_string()),
            token_ids: vec![0; num_tokens],
        }],
    });
    with_tenant_and_cell(ctx)
}

/// A classification request after its preparation, which it shares with
/// embeddings: `num_tokens` input tokens.
fn classify_ctx(components: Arc<SharedComponents>, num_tokens: usize) -> RequestContext {
    let request: ClassifyRequest =
        serde_json::from_value(json!({"model": MODEL, "input": "Hello"})).unwrap();
    let mut ctx =
        RequestContext::for_classify(Arc::new(request), None, MODEL.to_string(), components);
    ctx.state.preparation = Some(PreparationOutput::Embedding {
        inputs: vec![EmbeddingInput {
            text: Some("Hello".to_string()),
            token_ids: vec![0; num_tokens],
        }],
    });
    with_tenant_and_cell(ctx)
}

/// A transcription request after its preparation: the chat-shaped
/// request the family's convention made, rendered to `num_tokens` tokens.
fn transcription_ctx(components: Arc<SharedComponents>, num_tokens: usize) -> RequestContext {
    let request = TranscriptionRequest {
        model: MODEL.to_string(),
        ..Default::default()
    };
    let audio = AudioFile {
        bytes: Bytes::from_static(b"not audio"),
        file_name: "clip.wav".to_string(),
        content_type: None,
    };
    let chat_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}]
    }))
    .unwrap();
    let mut ctx = RequestContext::for_transcription(
        Arc::new(request),
        Arc::new(audio),
        None,
        MODEL.to_string(),
        components,
    );
    ctx.state.preparation = Some(PreparationOutput::Transcription {
        token_ids: vec![0; num_tokens],
        processed_messages: ProcessedMessages {
            text: String::new(),
            unbilled_prompt_tokens: 0,
            continued_final_message: false,
            generation_prompt: true,
        },
        chat_request: Arc::new(chat_request),
        format: TranscriptionResponseFormat::Json,
        family: FAMILIES[0],
    });
    with_tenant_and_cell(ctx)
}

/// The reservation of `endpoint`, as the chat endpoint takes it: the
/// estimate is the preparation's input token count; a budget that cannot
/// hold it refuses with the chat endpoint's 429, a budget that holds it
/// once admits the first request and refuses the second.
async fn assert_reserves_like_chat(
    endpoint: &str,
    make: fn(Arc<SharedComponents>, usize) -> RequestContext,
) {
    let components = test_components().await;

    // Too small for one 5-token reservation.
    let stage = RateLimitReserveStage::new(Some(manager_with_tokens_per_minute(3)));
    let mut ctx = make(components.clone(), 5);
    let response = stage.execute(&mut ctx).await.expect_err(endpoint);
    assert_eq!(
        response.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "{endpoint}"
    );
    assert!(
        matches!(
            ctx.input.rate_limit_cell.as_ref().unwrap().peek(),
            Some(RateLimitOutcome::Denied)
        ),
        "{endpoint}: the denial is cached on the cell"
    );

    // Room for one 5-token reservation, not two (5 + 5 > 6).
    let stage = RateLimitReserveStage::new(Some(manager_with_tokens_per_minute(6)));
    let mut first = make(components.clone(), 5);
    assert!(stage.execute(&mut first).await.is_ok(), "{endpoint}");
    assert!(
        matches!(
            first.input.rate_limit_cell.as_ref().unwrap().peek(),
            Some(RateLimitOutcome::Admitted(_))
        ),
        "{endpoint}: the admission is cached on the cell"
    );
    let mut second = make(components, 5);
    let response = stage.execute(&mut second).await.expect_err(endpoint);
    assert_eq!(
        response.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "{endpoint}: the first request's tokens are debited"
    );
}

#[tokio::test]
async fn an_embeddings_request_reserves_its_input_tokens_like_a_chat_request() {
    assert_reserves_like_chat("embeddings", embedding_ctx).await;
}

#[tokio::test]
async fn a_classify_request_reserves_its_input_tokens_like_a_chat_request() {
    assert_reserves_like_chat("classify", classify_ctx).await;
}

#[tokio::test]
async fn a_transcription_request_reserves_its_prompt_tokens_like_a_chat_request() {
    assert_reserves_like_chat("transcription", transcription_ctx).await;
}

/// Embeddings and classification settle the prompt tokens the engine
/// counted and nothing else.
#[test]
fn prompt_only_settlement_charges_the_prompt_alone() {
    assert_eq!(
        RequestPipeline::prompt_only_settlement(7),
        UsageSettlement {
            actual_input_tokens: 7,
            completion_tokens: 0,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        }
    );
}
