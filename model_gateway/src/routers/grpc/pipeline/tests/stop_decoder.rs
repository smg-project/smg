//! The stop sequence decoder of a collected answer is built by preparation for a
//! collected answer alone: a stream decodes each choice as it reads and gets none.

use openai_protocol::{
    chat::ChatCompletionRequest, generate::GenerateRequest, messages::CreateMessageRequest,
};

use super::*;

fn request<T: serde::de::DeserializeOwned>(value: serde_json::Value) -> Arc<T> {
    Arc::new(serde_json::from_value(value).expect("a request"))
}

#[tokio::test]
async fn preparation_builds_the_stop_decoder_for_a_collected_answer_only() {
    let worker_registry = Arc::new(WorkerRegistry::new());
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    );
    let components = components(worker_registry).await;
    for stream in [false, true] {
        let chat: Arc<ChatCompletionRequest> = request(serde_json::json!({
            "model": MODEL, "stream": stream, "stop": ["\n"],
            "messages": [{"role": "user", "content": "Hello"}]
        }));
        let generate: Arc<GenerateRequest> = request(serde_json::json!({
            "model": MODEL, "stream": stream, "text": "Hello",
            "sampling_params": {"stop": ["\n"]}
        }));
        let completion: Arc<CompletionRequest> = request(serde_json::json!({
            "model": MODEL, "stream": stream, "stop": ["\n"], "prompt": "Hello"
        }));
        let messages: Arc<CreateMessageRequest> = request(serde_json::json!({
            "model": MODEL, "stream": stream, "stop_sequences": ["\n"], "max_tokens": 8,
            "messages": [{"role": "user", "content": "Hello"}]
        }));
        let model = || MODEL.to_string();
        let contexts = [
            (
                "chat",
                Endpoint::Chat,
                RequestContext::for_chat(chat, None, model(), components.clone()),
            ),
            (
                "generate",
                Endpoint::Chat,
                RequestContext::for_generate(generate, None, model(), components.clone()),
            ),
            (
                "completion",
                Endpoint::Completion,
                RequestContext::for_completion(completion, None, model(), components.clone()),
            ),
            (
                "messages",
                Endpoint::Messages,
                RequestContext::for_messages(messages, None, model(), components.clone()),
            ),
        ];
        for (name, endpoint, mut ctx) in contexts {
            let pipeline =
                RequestPipeline::build(endpoint, Mode::Regular, &deps).expect("a pipeline");
            pipeline
                .stages
                .preparation
                .execute(&mut ctx)
                .await
                .unwrap_or_else(|_| panic!("{name} prepares, stream {stream}"));
            assert_eq!(
                ctx.state.response.stop_decoder.is_some(),
                !stream,
                "{name}, stream {stream}"
            );
        }
    }
}
