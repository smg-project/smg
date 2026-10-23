use std::collections::HashMap;

use llm_multimodal::{
    EncoderFieldLayouts, FieldLayout, ImageFrame, ImageSource, PlaceholderRange,
    PreprocessedEncoderInputs,
};
use llm_tokenizer::TokenizerRegistry;
use ndarray::{ArrayD, IxDyn};
use smg_grpc_client::SglangSchedulerClient;

use super::*;
use crate::{
    routers::grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        common::stages::{encode::EncodeStage, helpers, PipelineStage},
        context::{
            ClientSelection, EncodeOutputs, RequestContext, SharedComponents, WorkerSelection,
        },
        multimodal::{MediaBatch, PrecomputedMultimodalIntermediate, PromptBinding},
        proto_wrapper::{MultimodalData, ProtoGenerateRequest},
        regular::symphony::template::Templates,
        utils::ParserResolver,
    },
    worker::{BasicWorkerBuilder, Worker, WorkerRegistry},
};

fn intermediate() -> MultimodalIntermediate {
    let images = super::media::originals();
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
    intermediate.sglang_encoder_media = Some(MediaPlan::new(
        images.into_iter().map(super::media::image_uri),
    ));
    intermediate
}

#[tokio::test]
async fn epd_sglang_actual_stage_and_build_keep_inline_media_and_final_id() {
    let (url, server) = crate::routers::grpc::test_engine::mock_engine("stage-model").await;
    let client = BackendClient::Grpc(GrpcClient::Sglang(
        SglangSchedulerClient::connect(&url).await.unwrap(),
    ));
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let components = Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(tokenizer_registry.clone())),
        tokenizer_registry,
        worker_registry: Arc::new(WorkerRegistry::new()),
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
    let role_worker = |role| {
        Arc::new(
            BasicWorkerBuilder::new(&url)
                .worker_type(role)
                .runtime_type(RuntimeType::Sglang)
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        ) as Arc<dyn Worker>
    };
    let a = super::media::encoder("grpc://encoder-a:5001");
    let b = super::media::encoder("grpc://encoder-b:5002");
    ctx.state.workers = Some(WorkerSelection::Disaggregated {
        encode_assignments: Some(
            [a.clone(), b.clone(), a, b]
                .into_iter()
                .enumerate()
                .map(|(item_index, worker)| EncodeWorkerAssignment { item_index, worker })
                .collect(),
        ),
        prefill: role_worker(WorkerType::Prefill),
        decode: role_worker(WorkerType::Decode),
        runtime_type: RuntimeType::Sglang,
    });
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
    let expected = super::media::originals();
    assert_eq!(data.inline_bytes(), expected.iter().map(String::len).sum());
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
    helpers::finish_generate_request(&mut ctx, &mut request, false, "epd_sglang_test").unwrap();
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
    assert_eq!(handoff.num_items_assigned, [2, 2]);
    assert!(
        matches!(&ctx.state.encode_outputs, Some(EncodeOutputs::Sglang(_))),
        "retry owns the immutable plan"
    );
    server.abort();
}
