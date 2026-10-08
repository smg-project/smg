//! The `Embed` RPC over the EngineCore wire: a pooling request in place of a
//! sampling one, answered by the engine's single finished output.
//!
//! vLLM's frontend (`AsyncLLM.encode` through `InputProcessor`) is bypassed
//! here, so what it would do to `PoolingParams(task="embed")` before the
//! engine sees it is what [`PoolingParams::embed`] encodes; the prompt-length
//! check it performs is repeated in [`translate_embed_request`].

use engine_zmq_client::protocol::vllm::{
    output::EngineCoreFinishReason,
    pooling::{PoolerDefaults, PoolingParams, POOLING_TASKS},
    request::EngineCoreRequest,
};
use futures::StreamExt;
use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;

use crate::{
    client::{zmq_status, ZmqEngineClient},
    vllm::{finish_reason_str, now_secs},
};

/// Translate a vLLM-proto embed request into a pooling `EngineCoreRequest`:
/// `pooling` in place of sampling params. ZMQ mode requires pre-tokenized
/// input (the Router tokenizes upstream); `max_model_len` bounds the prompt as
/// vLLM's own input processor does.
pub fn translate_embed_request(
    req: vllm::EmbedRequest,
    pooling: PoolingParams,
    max_model_len: u64,
) -> Result<EngineCoreRequest, String> {
    if req.request_id.is_empty() {
        return Err("request_id is required".to_string());
    }
    let Some(tokenized) = req.tokenized else {
        return Err("EmbedRequest requires tokenized input".to_string());
    };
    if tokenized.input_ids.is_empty() {
        return Err("the prompt cannot be empty".to_string());
    }
    let prompt_len = tokenized.input_ids.len() as u64;
    if prompt_len > max_model_len {
        return Err(format!(
            "the prompt (length {prompt_len}) is longer than the maximum model length of \
             {max_model_len}"
        ));
    }
    Ok(EngineCoreRequest {
        request_id: req.request_id,
        prompt_token_ids: Some(tokenized.input_ids),
        sampling_params: None,
        pooling_params: Some(pooling),
        arrival_time: now_secs(),
        ..EngineCoreRequest::default()
    })
}

impl ZmqEngineClient {
    /// Submit a vLLM-proto embed request to a vLLM backend and await its
    /// pooled vector. A pooling request has no sampling duties for the
    /// frontend; the engine answers it with exactly one finished output,
    /// which becomes the `EmbedResponse` (`prompt_tokens` is the prompt's
    /// length, as the Python servicer reports it). `pooler` is the model's
    /// pooler config, merged into the params as vLLM's frontend merges it.
    pub async fn embed(
        &self,
        req: vllm::EmbedRequest,
        pooler: PoolerDefaults,
    ) -> Result<vllm::EmbedResponse, Status> {
        let client = self.vllm_client("Embed")?;
        let max_model_len = client
            .engines()
            .first()
            .map(|engine| engine.ready_response.max_model_len)
            .ok_or_else(|| Status::unavailable("no connected ZMQ engine"))?;
        let request = translate_embed_request(req, PoolingParams::embed_for(pooler), max_model_len)
            .map_err(Status::invalid_argument)?;
        // vLLM's frontend refuses a task the model runner does not serve
        // (`InputProcessor._validate_params`, in its words); the engine's own
        // check raises on its busy loop and takes EngineCore down with it.
        let tasks = self.supported_tasks().await?;
        let pooling_tasks: Vec<&str> = tasks
            .iter()
            .map(String::as_str)
            .filter(|task| POOLING_TASKS.contains(task))
            .collect();
        if pooling_tasks.is_empty() {
            return Err(Status::invalid_argument(
                "This model does not support pooling",
            ));
        }
        if !pooling_tasks.contains(&"embed") {
            return Err(Status::invalid_argument(format!(
                "Unsupported task: 'embed' Supported tasks: {pooling_tasks:?}"
            )));
        }
        let request_id = request.request_id.clone();
        let prompt_tokens = request
            .prompt_token_ids
            .as_ref()
            .map_or(0, |ids| u32::try_from(ids.len()).unwrap_or(u32::MAX));
        // Dropped before the finishing output (an error below, or the caller
        // giving up), the stream aborts the engine-side request.
        let mut stream = client.submit(request).await.map_err(zmq_status)?;
        while let Some(output) = stream.next().await {
            let output = output.map_err(zmq_status)?;
            // Pooling finishes on the tick that produces the output; a tick
            // before it carries nothing for this request.
            let Some(reason) = output.finish_reason else {
                continue;
            };
            return match (reason, output.pooling_output) {
                (EngineCoreFinishReason::Stop, Some(pooled)) => {
                    let embedding = pooled.to_vector().map_err(|error| {
                        Status::internal(format!(
                            "pooling output of request {request_id} could not be decoded: {error}"
                        ))
                    })?;
                    Ok(vllm::EmbedResponse {
                        embedding_dim: u32::try_from(embedding.len()).unwrap_or(u32::MAX),
                        embedding,
                        prompt_tokens,
                    })
                }
                (EngineCoreFinishReason::Abort, _) => Err(Status::aborted(format!(
                    "embed request {request_id} was aborted"
                ))),
                (reason, _) => Err(Status::internal(format!(
                    "embed request {request_id} finished ({}) without a pooling output",
                    finish_reason_str(reason)
                ))),
            };
        }
        Err(Status::internal(format!(
            "Embed request {request_id} did not produce a result"
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, time::Duration};

    use engine_zmq_client::{
        codec::{tensor::WireTensor, OpaqueValue},
        mock_engine::{
            connect_to_frontend, default_ready_response, EngineInbound, MockEngineInput,
            MockEngineOutput,
        },
        protocol::vllm::{
            output::{EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs},
            pooling::PoolingOutput,
        },
        EngineId,
    };
    use openai_protocol::worker::RuntimeType;
    use tonic::Code;

    use super::*;
    use crate::eos::EosTokenIds;

    fn embed_request(id: &str, input_ids: Vec<u32>) -> vllm::EmbedRequest {
        vllm::EmbedRequest {
            request_id: id.to_string(),
            tokenized: Some(vllm::TokenizedInput {
                original_text: String::new(),
                input_ids,
            }),
        }
    }

    #[test]
    fn translate_requires_a_tokenized_prompt_within_the_context() {
        let params = PoolingParams::embed;
        let untokenized = vllm::EmbedRequest {
            request_id: "e".to_string(),
            tokenized: None,
        };
        let error = translate_embed_request(untokenized, params(), 512).unwrap_err();
        assert!(error.contains("tokenized"), "{error}");
        let error =
            translate_embed_request(embed_request("e", Vec::new()), params(), 512).unwrap_err();
        assert!(error.contains("empty"), "{error}");
        let error =
            translate_embed_request(embed_request("e", vec![1, 2, 3]), params(), 2).unwrap_err();
        assert!(error.contains("maximum model length"), "{error}");
        let error = translate_embed_request(embed_request("", vec![1]), params(), 2).unwrap_err();
        assert!(error.contains("request_id"), "{error}");

        // A prompt filling the whole context is valid: nothing is generated.
        let request = translate_embed_request(embed_request("e", vec![1, 2, 3]), params(), 3)
            .expect("translates");
        assert_eq!(request.request_id, "e");
        assert_eq!(request.prompt_token_ids, Some(vec![1, 2, 3]));
        assert!(request.sampling_params.is_none());
        assert_eq!(request.pooling_params, Some(PoolingParams::embed()));
    }

    /// One finished engine output for `request_id`, with or without a
    /// pooled vector.
    fn pooled(
        request_id: &str,
        finish: EngineCoreFinishReason,
        values: Option<Vec<f32>>,
    ) -> EngineCoreOutputs {
        let pooling_output = values.map(|values| {
            PoolingOutput::new(WireTensor::from_f32(vec![values.len()], values).unwrap())
        });
        EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: request_id.to_string(),
                pooling_output,
                finish_reason: Some(finish),
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from([request_id.to_string()])),
            ..Default::default()
        })
    }

    /// Answer the engine's one-time `get_supported_tasks` utility call.
    async fn answer_supported_tasks(
        engine_in: &mut MockEngineInput,
        engine_out: &mut MockEngineOutput,
        tasks: &[&str],
    ) {
        let EngineInbound::Utility(call) = engine_in.recv().await.unwrap() else {
            panic!("expected the supported-tasks utility call");
        };
        assert_eq!(call.method, "get_supported_tasks");
        assert!(call.args.is_empty());
        let tasks: Vec<OpaqueValue> = tasks.iter().map(|task| OpaqueValue::from(*task)).collect();
        engine_out
            .send_utility_reply(0, call.call_id, Ok(OpaqueValue::Array(tasks)))
            .await
            .unwrap();
    }

    async fn connected(
        dir: &std::path::Path,
    ) -> (ZmqEngineClient, MockEngineInput, MockEngineOutput) {
        let ep = |name: &str| format!("ipc://{}", dir.join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let (engine_in, engine_out) = engine.expect("mock engine").split();
        (client, engine_in, engine_out)
    }

    /// The engine's task list gates the request before anything reaches the
    /// scheduler: a generation runner has no pooling task, a classifier has
    /// pooling tasks but not `embed`. Fetched once per connection.
    #[tokio::test]
    async fn embed_is_refused_when_the_engine_lacks_the_task() {
        let dir = tempfile::tempdir().unwrap();
        let (client, mut engine_in, mut engine_out) = connected(dir.path()).await;
        let (status, ()) = tokio::join!(
            client.embed(embed_request("e1", vec![1]), PoolerDefaults::default()),
            answer_supported_tasks(&mut engine_in, &mut engine_out, &["generate"]),
        );
        let status = status.expect_err("refused");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(status.message(), "This model does not support pooling");
        // Cached: no second utility call, the refusal is immediate.
        let status = client
            .embed(embed_request("e2", vec![1]), PoolerDefaults::default())
            .await
            .expect_err("refused");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(
            tokio::time::timeout(Duration::from_millis(200), engine_in.recv())
                .await
                .is_err(),
            "nothing reached the engine"
        );

        let dir = tempfile::tempdir().unwrap();
        let (client, mut engine_in, mut engine_out) = connected(dir.path()).await;
        let (status, ()) = tokio::join!(
            client.embed(embed_request("e3", vec![1]), PoolerDefaults::default()),
            answer_supported_tasks(&mut engine_in, &mut engine_out, &["classify", "render"]),
        );
        let status = status.expect_err("refused");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(
            status.message(),
            "Unsupported task: 'embed' Supported tasks: [\"classify\"]"
        );
    }

    /// End-to-end over ipc://: the embed request reaches the engine as a
    /// pooling request, and the engine's finished output comes back as the
    /// vector; an engine-side abort and a finish without output are errors.
    #[tokio::test]
    async fn embed_e2e_submits_pooling_params_and_maps_the_vector() {
        let dir = tempfile::tempdir().unwrap();
        let (client, mut engine_in, mut engine_out) = connected(dir.path()).await;

        let pooler = PoolerDefaults {
            use_activation: Some(false),
            dimensions: Some(2),
        };
        let (response, request) = tokio::join!(
            client.embed(embed_request("e1", vec![101, 7592, 102]), pooler),
            async {
                answer_supported_tasks(&mut engine_in, &mut engine_out, &["embed", "classify"])
                    .await;
                let EngineInbound::Add(request) = engine_in.recv().await.unwrap() else {
                    panic!("expected Add");
                };
                engine_out
                    .send_outputs(&pooled(
                        "e1",
                        EngineCoreFinishReason::Stop,
                        Some(vec![0.5, -0.25]),
                    ))
                    .await
                    .unwrap();
                request
            }
        );
        assert_eq!(request.prompt_token_ids, Some(vec![101, 7592, 102]));
        assert!(request.sampling_params.is_none());
        // The pooler config rode into the params.
        assert_eq!(
            request.pooling_params,
            Some(PoolingParams::embed_for(pooler))
        );
        let response = response.expect("embed");
        assert_eq!(response.embedding, vec![0.5, -0.25]);
        assert_eq!(response.prompt_tokens, 3);
        assert_eq!(response.embedding_dim, 2);

        let (response, ()) = tokio::join!(
            client.embed(embed_request("e2", vec![1]), PoolerDefaults::default()),
            async {
                engine_in.recv().await.unwrap();
                engine_out
                    .send_outputs(&pooled("e2", EngineCoreFinishReason::Abort, None))
                    .await
                    .unwrap();
            }
        );
        assert_eq!(response.unwrap_err().code(), Code::Aborted);

        let (response, ()) = tokio::join!(
            client.embed(embed_request("e3", vec![1]), PoolerDefaults::default()),
            async {
                engine_in.recv().await.unwrap();
                engine_out
                    .send_outputs(&pooled("e3", EngineCoreFinishReason::Stop, None))
                    .await
                    .unwrap();
            }
        );
        let status = response.unwrap_err();
        assert_eq!(status.code(), Code::Internal);
        assert!(status.message().contains("without a pooling output"));
    }
}
