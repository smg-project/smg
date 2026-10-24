use std::collections::HashMap;

use axum::http::StatusCode;
use llm_multimodal::{
    EncoderFieldLayouts, FieldLayout, ImageFrame, ImageSource, PlaceholderRange,
    PreprocessedEncoderInputs,
};
use llm_tokenizer::TokenizerRegistry;
use ndarray::{ArrayD, IxDyn};
use smg_grpc_client::SglangSchedulerClient;

use super::*;
use crate::{
    config::types::PolicyConfig,
    policies::PolicyRegistry,
    routers::grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        common::stages::{
            encode::EncodeStage,
            helpers,
            worker_selection::{WorkerSelectionMode, WorkerSelectionStage},
            PipelineStage,
        },
        context::{
            ClientSelection, EncodeOutputs, PreparationOutput, RequestContext, SharedComponents,
        },
        multimodal::{MediaBatch, PrecomputedMultimodalIntermediate, PromptBinding},
        proto_wrapper::{MultimodalData, ProtoGenerateRequest},
        regular::symphony::template::Templates,
        utils::ParserResolver,
    },
    worker::{BasicWorkerBuilder, ModelCard, Worker, WorkerRegistry},
};

fn intermediate() -> MultimodalIntermediate {
    let images = media::originals();
    let frames = images
        .iter()
        .enumerate()
        .map(|(i, url)| {
            Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                STANDARD
                    .decode(url.split_once(',').unwrap().1)
                    .unwrap()
                    .into(),
                ImageDetail::Auto,
                ImageSource::DataUrl,
                format!("{i:064x}"),
            ))
        })
        .collect();
    // Vision computation is a stand-in; media, Stage, assembly and finalizer are real.
    let mut intermediate =
        MultimodalIntermediate::try_new(vec![PrecomputedMultimodalIntermediate {
            preprocessed: PreprocessedEncoderInputs {
                encoder_input: ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![0.0f32; 4])
                    .unwrap()
                    .into(),
                feature_token_counts: vec![1; 4],
                item_sizes: vec![(1, 1); 4],
                model_specific: HashMap::new(),
            },
            media: MediaBatch::Images(frames),
            bindings: (0..4)
                .map(|i| PromptBinding {
                    item_index: i,
                    prompt_ordinal: i,
                    structural: PlaceholderRange {
                        offset: i,
                        length: 1,
                    },
                    patches: vec![PlaceholderRange {
                        offset: i,
                        length: 1,
                    }],
                })
                .collect(),
            placeholder_token_id: Some(1),
            field_layouts: EncoderFieldLayouts::new(FieldLayout::Batched, HashMap::new()),
            keep_on_cpu_keys: Vec::new(),
            encoder_input_key: None,
        }])
        .unwrap();
    intermediate.sglang_encoder_media =
        Some(MediaPlan::new(images.into_iter().map(media::image_uri)));
    intermediate
}

#[tokio::test]
async fn epd_sglang_actual_stage_and_build_keep_inline_media_and_final_id() {
    let (url, server) = crate::routers::grpc::test_engine::mock_engine("stage-model").await;
    let client = BackendClient::Grpc(GrpcClient::Sglang(
        SglangSchedulerClient::connect(&url).await.unwrap(),
    ));
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let registry = Arc::new(WorkerRegistry::new());
    let components = Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(tokenizer_registry.clone())),
        tokenizer_registry,
        worker_registry: registry.clone(),
        tool_parser_factory: Default::default(),
        reasoning_parser_factory: Default::default(),
        parser_resolver: ParserResolver::disabled(),
        multimodal: None,
        tokenizer_autoload: false,
        tokenizer_load_failures: Arc::default(),
        parser: crate::config::ParserChoice::Legacy,
    });
    let input = serde_json::from_value(serde_json::json!({"text": "image prompt"})).unwrap();
    let mut ctx =
        RequestContext::for_generate(Arc::new(input), None, "stage-model".into(), components);
    let role_worker = |url: &str, role| {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("stage-model"))
                .worker_type(role)
                .runtime_type(RuntimeType::Sglang)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        ) as Arc<dyn Worker>
    };
    let first_encoder = role_worker("grpc://encoder-a:5001", WorkerType::Encode);
    for worker in [
        first_encoder.clone(),
        role_worker(&url, WorkerType::Prefill),
        role_worker("grpc://decode:5002", WorkerType::Decode),
    ] {
        registry.register(worker).unwrap();
    }
    let selection = WorkerSelectionStage::new(
        registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        WorkerSelectionMode::EncodePrefillDecode,
        None,
    );
    ctx.state.preparation = Some(PreparationOutput::Generate {
        original_text: Some("image prompt".into()),
        token_ids: vec![1, 2, 3, 4],
    });
    ctx.state.multimodal_intermediate = Some(intermediate());
    selection.execute(&mut ctx).await.unwrap();
    let assignments = ctx
        .state
        .workers
        .as_ref()
        .unwrap()
        .encode_assignments()
        .unwrap();
    assert_eq!(
        assignments.len(),
        4,
        "initial selection read real nonempty media hashes"
    );
    assert!(assignments
        .iter()
        .all(|a| a.worker.url() == first_encoder.url()));
    ctx.state.clients = Some(ClientSelection::Disaggregated {
        prefill: client.clone(),
        decode: client,
    });
    ctx.state.multimodal_intermediate = Some(intermediate());
    EncodeStage::new().execute(&mut ctx).await.unwrap();
    assert!(matches!(
        &ctx.state.encode_outputs,
        Some(EncodeOutputs::Sglang(_))
    ));
    let data = helpers::assemble_prompt_media(&mut ctx, "epd_sglang_test")
        .await
        .unwrap()
        .unwrap();
    let expected = media::originals();
    assert_eq!(
        data.inline_bytes(),
        expected.iter().map(String::len).sum::<usize>()
    );
    let MultimodalData::Sglang(data) = data else {
        panic!("wrong backend")
    };
    let mut request = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
        request_id: "built-final-id".into(),
        mm_inputs: Some(data.into_proto()),
        sampling_params: Some(sglang_proto::SamplingParams {
            n: 1,
            ..Default::default()
        }),
        ..Default::default()
    }));
    helpers::finish_generate_request(
        &mut ctx,
        &mut request,
        crate::routers::grpc::mode::Mode::EncodePrefillDecode.inject_pd_metadata(),
        "epd_sglang_test",
    )
    .unwrap();
    let bootstrap = request.as_sglang().disaggregated_params.as_ref().unwrap();
    let prefill = ctx
        .state
        .workers
        .as_ref()
        .unwrap()
        .prefill_worker()
        .unwrap();
    assert_eq!(
        bootstrap.bootstrap_host,
        prefill.metadata().bootstrap_host()
    );
    assert_eq!(
        bootstrap.bootstrap_port,
        prefill
            .metadata()
            .bootstrap_port()
            .unwrap_or(crate::worker::DEFAULT_BOOTSTRAP_PORT) as i32
    );
    assert!(bootstrap.bootstrap_room >= 0);
    assert_eq!(
        request.as_sglang().mm_inputs.as_ref().unwrap().image_urls,
        expected
    );
    assert!(request
        .as_sglang()
        .mm_inputs
        .as_ref()
        .unwrap()
        .pixel_values
        .is_none());
    let handoff = request.as_sglang().encode_handoff.as_ref().unwrap();
    assert_eq!(handoff.request_id, "built-final-id");
    assert_eq!(handoff.num_items_assigned, [4]);
    assert!(
        matches!(&ctx.state.encode_outputs, Some(EncodeOutputs::Sglang(_))),
        "retry owns the immutable plan"
    );
    // A retry reselects all three SG pools, including a replacement encoder.
    let mut retry = ctx.into_dispatch().unwrap();
    first_encoder.set_status(openai_protocol::worker::WorkerStatus::NotReady);
    let replacement = role_worker("grpc://encoder-b:5001", WorkerType::Encode);
    registry.register(replacement.clone()).unwrap();
    selection.reselect(&mut retry).await.unwrap();
    let assignments = retry
        .workers
        .as_ref()
        .unwrap()
        .encode_assignments()
        .unwrap();
    assert_eq!(assignments.len(), 4);
    assert!(assignments
        .iter()
        .all(|a| a.worker.url() == replacement.url()));
    assert_eq!(
        retry
            .workers
            .as_ref()
            .unwrap()
            .prefill_worker()
            .unwrap()
            .metadata()
            .spec
            .runtime_type,
        RuntimeType::Sglang
    );
    replacement.set_status(openai_protocol::worker::WorkerStatus::NotReady);
    assert_eq!(
        selection.reselect(&mut retry).await.unwrap_err().status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "no silent PD fallback without encoder"
    );
    server.abort();
}
