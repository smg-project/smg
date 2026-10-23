//! A request's `no_stop_trim` (keep the matched stop string in the output)
//! reaches every engine's own field for it, on the plain and on the chat path,
//! and stays unset when the request does not ask.

use openai_protocol::{chat::ChatCompletionRequest, generate::GenerateRequest};
use smg_grpc_client::{
    sglang_scheduler::{SglangGenerateRequestOptions, SglangSchedulerClient},
    tokenspeed_scheduler::TokenSpeedSchedulerClient,
    trtllm_service::TrtllmServiceClient,
    vllm_engine::VllmEngineClient,
};

fn plain(no_stop_trim: bool) -> serde_json::Result<GenerateRequest> {
    serde_json::from_value(serde_json::json!({
        "text": "hi",
        "sampling_params": {"stop": ["\n"], "no_stop_trim": no_stop_trim}
    }))
}

#[test]
fn a_plain_requests_no_stop_trim_reaches_every_engine_client() {
    for asked in [true, false] {
        let request = plain(asked).unwrap();
        let vllm =
            VllmEngineClient::build_plain_generate_request("r".into(), &request, None, vec![1])
                .unwrap();
        assert_eq!(
            vllm.sampling_params.unwrap().include_stop_str_in_output,
            asked,
            "vllm_engine, asked {asked}"
        );
        let sglang = SglangSchedulerClient::build_plain_generate_request_parts(
            "r".into(),
            &request,
            None,
            vec![1],
        )
        .unwrap();
        assert_eq!(
            sglang.sampling_params.unwrap().no_stop_trim,
            asked,
            "sglang_scheduler, asked {asked}"
        );
        let tokenspeed = TokenSpeedSchedulerClient::build_plain_generate_request(
            "r".into(),
            &request,
            None,
            vec![1],
        )
        .unwrap();
        assert_eq!(
            tokenspeed.sampling_params.unwrap().no_stop_trim,
            asked,
            "tokenspeed_scheduler, asked {asked}"
        );
        let trtllm = TrtllmServiceClient::build_plain_generate_request_parts(
            "r".into(),
            &request,
            None,
            vec![1],
        )
        .unwrap();
        assert_eq!(
            trtllm.include_stop_token_in_output, asked,
            "trtllm_service, asked {asked}"
        );
    }
}

fn chat(no_stop_trim: bool) -> serde_json::Result<ChatCompletionRequest> {
    serde_json::from_value(serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": ["\n"],
        "no_stop_trim": no_stop_trim
    }))
}

#[test]
fn a_chat_requests_no_stop_trim_reaches_every_engine_client() {
    for asked in [true, false] {
        let request = chat(asked).unwrap();
        let vllm = VllmEngineClient::build_generate_request_from_chat(
            "r".into(),
            &request,
            "hi".into(),
            vec![1],
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            vllm.sampling_params.unwrap().include_stop_str_in_output,
            asked,
            "vllm_engine, asked {asked}"
        );
        let sglang = SglangSchedulerClient::build_generate_request_from_chat_parts(
            "r".into(),
            &request,
            "hi".into(),
            vec![1],
            SglangGenerateRequestOptions::default(),
        )
        .unwrap();
        assert_eq!(
            sglang.sampling_params.unwrap().no_stop_trim,
            asked,
            "sglang_scheduler, asked {asked}"
        );
        let tokenspeed = TokenSpeedSchedulerClient::build_generate_request_from_chat(
            "r".into(),
            &request,
            "hi".into(),
            vec![1],
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            tokenspeed.sampling_params.unwrap().no_stop_trim,
            asked,
            "tokenspeed_scheduler, asked {asked}"
        );
        let trtllm = TrtllmServiceClient::build_generate_request_from_chat_parts(
            "r".into(),
            &request,
            "hi".into(),
            vec![1],
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            trtllm.include_stop_token_in_output, asked,
            "trtllm_service, asked {asked}"
        );
    }
}
