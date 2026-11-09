//! The two HTTP mock engines, the harness's (`common::mock_worker`) and the mock
//! worker crate's (`mock_worker::http`), answer nine routes in common. This test
//! drives every one of them on both engines with the same request and pins the
//! shape of the two answers equal: the status, the content type, and the keys
//! and value kinds four levels into the body (a chat choice's message included). Where the shapes differ today the
//! difference is listed in [`DIFFERENCES`]; a route that moves from one engine to
//! the other closes its row or keeps it knowingly, and a new difference fails here
//! with the table to paste.
#![expect(clippy::unwrap_used, reason = "test failures should panic")]

use std::{collections::BTreeMap, sync::Arc};

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use mock_worker::{config::Config, http::serve_with_listener_until};
use reqwest::{header::CONTENT_TYPE, Client, Method};
use serde_json::{json, Value};
use tokio::{net::TcpListener, sync::oneshot};

use crate::common;

/// The nine routes both engines serve: method, path, body.
fn shared_routes() -> Vec<(Method, &'static str, Option<Value>)> {
    vec![
        (Method::GET, "/health", None),
        (Method::GET, "/server_info", None),
        (Method::GET, "/get_server_info", None),
        (Method::GET, "/v1/models", None),
        (Method::GET, "/v1/loads", None),
        (Method::POST, "/flush_cache", Some(json!({}))),
        (
            Method::POST,
            "/generate",
            Some(json!({"text": "hello", "sampling_params": {"max_new_tokens": 4}})),
        ),
        (
            Method::POST,
            "/v1/chat/completions",
            Some(json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4
            })),
        ),
        (
            Method::POST,
            "/v1/completions",
            Some(json!({"model": "mock-model", "prompt": "hello", "max_tokens": 4})),
        ),
    ]
}

/// The differences the two engines have today, route by route: the harness's
/// shape, then the crate's; the three completion routes (chat, text and
/// generate) are the ones of the nine they answer alike. The crate answers
/// `/health` as text where the harness answers JSON; the loads, models, server
/// info and flush_cache bodies carry different key sets. A route that moves
/// engines closes its row here.
const DIFFERENCES: &[(&str, &str, &str)] = &[
    ("GET /get_server_info", "200 application/json {chat_template:null,context_length:num,device:str,disable_radix_cache:bool,dp_size:num,dtype:str,enable_flashinfer:bool,enable_p2p_check:bool,enable_torch_compile:bool,host:str,internal_states:[{running_queue_size:num,waiting_queue_size:num}],max_batch_tokens:num,max_num_batched_tokens:num,max_prefill_tokens:num,max_req_num:num,max_running_requests:num,mem_fraction_static:num,min_running_requests:num,model_path:str,port:num,req_to_token_ratio:num,running_queue_size:num,schedule_conservativeness:num,schedule_policy:str,served_model_name:str,show_time_cost:bool,stream_interval:num,tokenizer_path:str,tp_size:num,trust_remote_code:bool,version:str,waiting_queue_size:num}", "200 application/json {model_path:str}"),
    ("GET /health", "200 application/json {status:str,timestamp:num,worker_type:str}", "200 text/plain text"),
    ("GET /server_info", "200 application/json {chat_template:null,context_length:num,device:str,disable_radix_cache:bool,dp_size:num,dtype:str,enable_flashinfer:bool,enable_p2p_check:bool,enable_torch_compile:bool,host:str,internal_states:[{running_queue_size:num,waiting_queue_size:num}],max_batch_tokens:num,max_num_batched_tokens:num,max_prefill_tokens:num,max_req_num:num,max_running_requests:num,mem_fraction_static:num,min_running_requests:num,model_path:str,port:num,req_to_token_ratio:num,running_queue_size:num,schedule_conservativeness:num,schedule_policy:str,served_model_name:str,show_time_cost:bool,stream_interval:num,tokenizer_path:str,tp_size:num,trust_remote_code:bool,version:str,waiting_queue_size:num}", "200 application/json {model_path:str}"),
    ("GET /v1/loads", "200 application/json {aggregate:{avg_token_usage:num,total_reqs:num,total_running_reqs:num,total_waiting_reqs:num},dp_rank_count:num,loads:[{dp_rank:num,max_total_num_tokens:num,num_running_reqs:num,num_used_tokens:num,num_waiting_reqs:num,token_usage:num}]}", "200 application/json {dp_rank_count:num,loads:[{cache_hit_rate:num,dp_rank:num,gen_throughput:num,max_running_requests:num,max_total_num_tokens:num,num_running_reqs:num,num_total_reqs:num,num_used_tokens:num,num_waiting_reqs:num,num_waiting_uncached_tokens:num,token_usage:num,utilization:num}],timestamp:str}"),
    ("GET /v1/models", "200 application/json {data:[{created:num,id:str,object:str,owned_by:str}],object:str}", "200 application/json {data:[{created:num,id:str,max_model_len:num,object:str,owned_by:str,root:str}],object:str}"),
    ("POST /flush_cache", "200 application/json {message:str}", "200 application/json {message:str,status:str}"),
];

/// The shape of a JSON value: keys sorted, value kinds, `depth` levels deep, the
/// first element standing for an array.
fn signature(value: &Value, depth: usize) -> String {
    match value {
        Value::Object(map) => {
            if depth == 0 {
                return "{..}".to_string();
            }
            let mut fields: Vec<String> = map
                .iter()
                .map(|(key, value)| format!("{key}:{}", signature(value, depth - 1)))
                .collect();
            fields.sort();
            format!("{{{}}}", fields.join(","))
        }
        Value::Array(items) => match items.first() {
            Some(first) if depth > 0 => format!("[{}]", signature(first, depth - 1)),
            Some(_) => "[..]".to_string(),
            None => "[]".to_string(),
        },
        Value::String(_) => "str".to_string(),
        Value::Number(_) => "num".to_string(),
        Value::Bool(_) => "bool".to_string(),
        Value::Null => "null".to_string(),
    }
}

/// One engine's answer to one route, reduced to its shape.
#[expect(clippy::print_stderr, reason = "the bodies themselves, when asked for")]
async fn observe(
    client: &Client,
    base: &str,
    method: &Method,
    path: &str,
    body: Option<&Value>,
) -> String {
    let mut request = client.request(method.clone(), format!("{base}{path}"));
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or("").trim().to_string())
        .unwrap_or_default();
    let text = response.text().await.unwrap();
    if std::env::var_os("CONTRACT_DUMP").is_some() {
        // The bodies themselves, for the mapping a route's move needs:
        // `CONTRACT_DUMP=1 cargo test -p smg --test http_tests -- mock_engines_contract --nocapture`.
        eprintln!("{method} {path} on {base}: {status} {text}");
    }
    let shape = match serde_json::from_str::<Value>(&text) {
        Ok(value) => signature(&value, 4),
        Err(_) if text.is_empty() => "empty".to_string(),
        Err(_) => "text".to_string(),
    };
    format!("{status} {content_type} {shape}")
}

/// The crate's HTTP engine on an ephemeral port, stopped when the sender drops.
#[expect(
    clippy::disallowed_methods,
    reason = "a test server that ends with the test"
)]
async fn start_crate_engine() -> (String, oneshot::Sender<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (stop, stopped) = oneshot::channel::<()>();
    let config = Arc::new(Config::default());
    tokio::spawn(serve_with_listener_until(config, listener, async move {
        let _ = stopped.await;
    }));
    (format!("http://127.0.0.1:{port}"), stop)
}

#[tokio::test]
async fn the_two_mock_engines_answer_the_shared_routes_in_one_shape() {
    let mut harness = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let harness_url = harness.start().await.unwrap();
    let (crate_url, _stop) = start_crate_engine().await;
    let client = smg::server::http_client();

    let mut differences: BTreeMap<String, (String, String)> = BTreeMap::new();
    for (method, path, body) in shared_routes() {
        let on_harness = observe(&client, &harness_url, &method, path, body.as_ref()).await;
        let on_crate = observe(&client, &crate_url, &method, path, body.as_ref()).await;
        if on_harness != on_crate {
            differences.insert(format!("{method} {path}"), (on_harness, on_crate));
        }
    }

    let expected: BTreeMap<String, (String, String)> = DIFFERENCES
        .iter()
        .map(|(route, harness, krate)| {
            (route.to_string(), (harness.to_string(), krate.to_string()))
        })
        .collect();
    let table: Vec<String> = differences
        .iter()
        .map(|(route, (harness, krate))| format!("    (\"{route}\", \"{harness}\", \"{krate}\"),"))
        .collect();
    assert_eq!(
        differences,
        expected,
        "the two engines' shapes differ from the pinned table; the table as observed:\n{}",
        table.join("\n")
    );
}
