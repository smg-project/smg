use smg::worker::metrics_aggregator::{aggregate_metrics, MetricPack};

#[test]
fn test_aggregate_simple() {
    let pack1 = MetricPack {
        labels: vec![("source".to_string(), "worker1".to_string())],
        metrics_text: r#"
# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{method="post",code="200"} 1027
http_requests_total{method="post",code="400"} 3
"#
        .to_string(),
    };
    let pack2 = MetricPack {
        labels: vec![("source".to_string(), "worker2".to_string())],
        metrics_text: r#"
# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{method="post",code="200"} 500
"#
        .to_string(),
    };

    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = r#"# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{code="200",method="post",source="worker1"} 1027
http_requests_total{code="400",method="post",source="worker1"} 3
http_requests_total{code="200",method="post",source="worker2"} 500
"#;
    assert_eq!(result.trim(), expected.trim());
}

#[test]
fn test_aggregate_multiple_metrics() {
    let pack1 = MetricPack {
        labels: vec![("source".to_string(), "w1".to_string())],
        metrics_text: r#"
# TYPE metric_a gauge
metric_a{dim="x"} 1.0
# TYPE metric_b_total counter
metric_b_total 10
"#
        .to_string(),
    };
    let pack2 = MetricPack {
        labels: vec![("source".to_string(), "w2".to_string())],
        metrics_text: r#"
# TYPE metric_a gauge
metric_a{dim="y"} 2.0
"#
        .to_string(),
    };

    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = r#"# TYPE metric_a gauge
metric_a{dim="x",source="w1"} 1
metric_a{dim="y",source="w2"} 2

# TYPE metric_b_total counter
metric_b_total{source="w1"} 10
"#;
    assert_eq_sorted(&result, expected);
}

#[test]
fn test_colons_in_names_and_label_values_are_preserved() {
    // Engines export `sglang:`/`vllm:`-prefixed metrics; colons must survive
    // aggregation so the same queries work against an engine's /metrics and
    // the gateway's /engine_metrics.
    let pack = |addr: &str, value: u32| MetricPack {
        labels: vec![("worker_addr".to_string(), addr.to_string())],
        metrics_text: format!(
            r#"
# HELP sglang:num_running_reqs The number of running requests: now.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{{model_name="org/model:v1"}} {value}
"#
        ),
    };

    let result = aggregate_metrics(vec![
        pack("http://10.0.0.1:8000", 3),
        pack("http://10.0.0.2:8000", 12),
    ])
    .unwrap();
    let expected = r#"# HELP sglang:num_running_reqs The number of running requests: now.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="org/model:v1",worker_addr="http://10.0.0.1:8000"} 3
sglang:num_running_reqs{model_name="org/model:v1",worker_addr="http://10.0.0.2:8000"} 12
"#;
    assert_eq!(result.trim(), expected.trim());
}

#[test]
fn test_colons_in_label_names_are_sanitized_not_restored() {
    // A colon is legal in a metric name and in a label value, never in a
    // label name: restoring one there would make the whole page unscrapeable,
    // so a label name keeps the `_` the old sanitizer gave it.
    let pack = MetricPack {
        labels: vec![(
            "worker_addr".to_string(),
            "http://10.0.0.1:8000".to_string(),
        )],
        metrics_text: r#"
# HELP vllm:num_requests_running Running: now.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model:name="org/model:v1"} 3
"#
        .to_string(),
    };
    let result = aggregate_metrics(vec![pack]).unwrap();
    let expected = r#"# HELP vllm:num_requests_running Running: now.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="org/model:v1",worker_addr="http://10.0.0.1:8000"} 3
"#;
    assert_eq!(result.trim(), expected.trim());
}

#[test]
fn test_label_name_collisions_drop_the_family_not_the_page() {
    // `model:name` sanitizes to `model_name`; a sample carrying both would
    // render duplicate label names, which invalidates the whole scrape, so
    // that family is left out and the others still render.
    let pack = MetricPack {
        labels: vec![],
        metrics_text: r#"
# HELP a:collides Collides.
# TYPE a:collides gauge
a:collides{model:name="x",model_name="y"} 1
# HELP a:fine Fine.
# TYPE a:fine gauge
a:fine{model_name="y"} 2
"#
        .to_string(),
    };
    let result = aggregate_metrics(vec![pack]).unwrap();
    assert!(!result.contains("a:collides"), "{result}");
    assert!(result.contains("a:fine{model_name=\"y\"} 2"), "{result}");
}

#[test]
fn test_colon_and_underscore_twins_stay_distinct() {
    // `a:b` and `a_b` are different Prometheus metrics; replacing colons with
    // underscores merged them into one family. Leading and doubled colons are
    // valid metric-name characters too.
    let pack = MetricPack {
        labels: vec![("worker_addr".to_string(), "w1".to_string())],
        metrics_text: "
# TYPE sglang:token_usage gauge
sglang:token_usage 0.25
# TYPE sglang_token_usage gauge
sglang_token_usage 0.5
# TYPE :leading gauge
:leading 1
# TYPE job::rule gauge
job::rule 2
"
        .to_string(),
    };

    let result = aggregate_metrics(vec![pack]).unwrap();
    let expected = r#"# TYPE sglang:token_usage gauge
sglang:token_usage{worker_addr="w1"} 0.25

# TYPE sglang_token_usage gauge
sglang_token_usage{worker_addr="w1"} 0.5

# TYPE :leading gauge
:leading{worker_addr="w1"} 1

# TYPE job::rule gauge
job::rule{worker_addr="w1"} 2
"#;
    assert_eq_sorted(&result, expected);
}

#[test]
fn test_literal_colon_escape_is_not_rewritten() {
    // A fixed escape would turn these literal `xsmgcolon0z` strings into `:`.
    let pack = MetricPack {
        labels: vec![("worker".into(), "hostxsmgcolon0z:8000".into())],
        metrics_text: r#"# HELP sglang:testxsmgcolon0z A xsmgcolon0z literal.
# TYPE sglang:testxsmgcolon0z gauge
sglang:testxsmgcolon0z{value="a:xsmgcolon0z:b"} 1
"#
        .into(),
    };

    let result = aggregate_metrics(vec![pack]).unwrap();
    let expected = r#"# HELP sglang:testxsmgcolon0z A xsmgcolon0z literal.
# TYPE sglang:testxsmgcolon0z gauge
sglang:testxsmgcolon0z{value="a:xsmgcolon0z:b",worker="hostxsmgcolon0z:8000"} 1
"#;
    assert_eq!(result.trim(), expected.trim());
}

#[test]
fn test_empty_input() {
    let result = aggregate_metrics(vec![]).unwrap();
    assert_eq!(result, "");
}

#[test]
fn test_invalid_metrics_are_skipped() {
    let pack1 = MetricPack {
        labels: vec![("source".to_string(), "worker1".to_string())],
        metrics_text: "invalid metrics text".to_string(),
    };
    let pack2 = MetricPack {
        labels: vec![("source".to_string(), "worker2".to_string())],
        metrics_text: "# TYPE valid_metric gauge\nvalid_metric 123\n".to_string(),
    };
    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = r#"# TYPE valid_metric gauge
valid_metric{source="worker2"} 123
"#;
    assert_eq!(result.trim(), expected.trim());
}

#[test]
fn test_real() {
    let pack1 = MetricPack {
        labels: vec![("source".to_string(), "worker1".to_string())],
        // https://docs.sglang.io/references/production_metrics.html
        metrics_text: r#"# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.128902e+06
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.557572e+06
# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.28
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.007507552643049313
# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2.3518979474117756e+06
sglang:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
# HELP sglang:e2e_request_latency_seconds Histogram of End-to-end request latency in seconds
# TYPE sglang:e2e_request_latency_seconds histogram
sglang:e2e_request_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.116093850019932e+06
sglang:e2e_request_latency_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:e2e_request_latency_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="0.8",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
sglang:e2e_request_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 866964.5791549598
sglang:time_per_output_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 73.0
sglang:time_per_output_token_seconds_bucket{le="0.015",model_name="meta-llama/Llama-3.1-8B-Instruct"} 382.0
sglang:time_per_output_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
# HELP sglang:func_latency_seconds Function latency in seconds
# TYPE sglang:func_latency_seconds histogram
sglang:func_latency_seconds_sum{name="generate_request"} 4.514771912145079
sglang:func_latency_seconds_bucket{le="0.05",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.07500000000000001",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.1125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.16875",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="+Inf",name="generate_request"} 14007.0
sglang:func_latency_seconds_count{name="generate_request"} 14007.0
# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 162.0
# HELP sglang:num_used_tokens The number of used tokens
# TYPE sglang:num_used_tokens gauge
sglang:num_used_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct"} 123859.0
# HELP sglang:gen_throughput The generate throughput (token/s)
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="meta-llama/Llama-3.1-8B-Instruct"} 86.50814177726902
# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2826.0
"#.to_string(),
    };
    let pack2 = MetricPack {
        labels: vec![("source".to_string(), "worker2".to_string())],
        metrics_text: pack1.metrics_text.clone(),
    };
    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = r#"# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 0.28
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 0.28

# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.001"} 0
sglang:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.005"} 0
sglang:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.01"} 0
sglang:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="+Inf"} 11008
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 2351897.9474117756
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 11008
sglang:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.001"} 0
sglang:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.005"} 0
sglang:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.01"} 0
sglang:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="+Inf"} 11008
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 2351897.9474117756
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 11008

# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.005"} 1
sglang:time_per_output_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.01"} 73
sglang:time_per_output_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.015"} 382
sglang:time_per_output_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="+Inf"} 7400757
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 866964.5791549598
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 7400757
sglang:time_per_output_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.005"} 1
sglang:time_per_output_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.01"} 73
sglang:time_per_output_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.015"} 382
sglang:time_per_output_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="+Inf"} 7400757
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 866964.5791549598
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 7400757

# HELP sglang:func_latency_seconds Function latency in seconds
# TYPE sglang:func_latency_seconds histogram
sglang:func_latency_seconds_bucket{name="generate_request",source="worker1",le="0.05"} 14006
sglang:func_latency_seconds_bucket{name="generate_request",source="worker1",le="0.07500000000000001"} 14006
sglang:func_latency_seconds_bucket{name="generate_request",source="worker1",le="0.1125"} 14006
sglang:func_latency_seconds_bucket{name="generate_request",source="worker1",le="0.16875"} 14006
sglang:func_latency_seconds_bucket{name="generate_request",source="worker1",le="+Inf"} 14007
sglang:func_latency_seconds_sum{name="generate_request",source="worker1"} 4.514771912145079
sglang:func_latency_seconds_count{name="generate_request",source="worker1"} 14007
sglang:func_latency_seconds_bucket{name="generate_request",source="worker2",le="0.05"} 14006
sglang:func_latency_seconds_bucket{name="generate_request",source="worker2",le="0.07500000000000001"} 14006
sglang:func_latency_seconds_bucket{name="generate_request",source="worker2",le="0.1125"} 14006
sglang:func_latency_seconds_bucket{name="generate_request",source="worker2",le="0.16875"} 14006
sglang:func_latency_seconds_bucket{name="generate_request",source="worker2",le="+Inf"} 14007
sglang:func_latency_seconds_sum{name="generate_request",source="worker2"} 4.514771912145079
sglang:func_latency_seconds_count{name="generate_request",source="worker2"} 14007

# HELP sglang:num_used_tokens The number of used tokens
# TYPE sglang:num_used_tokens gauge
sglang:num_used_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 123859
sglang:num_used_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 123859

# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 0.007507552643049313
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 0.007507552643049313

# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 2826
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 2826

# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 7557572
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 7557572

# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 162
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 162

# HELP sglang:e2e_request_latency_seconds Histogram of End-to-end request latency in seconds
# TYPE sglang:e2e_request_latency_seconds histogram
sglang:e2e_request_latency_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.3"} 0
sglang:e2e_request_latency_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.5"} 6
sglang:e2e_request_latency_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="0.8"} 6
sglang:e2e_request_latency_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1",le="+Inf"} 11228
sglang:e2e_request_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 3116093.850019932
sglang:e2e_request_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 11228
sglang:e2e_request_latency_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.3"} 0
sglang:e2e_request_latency_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.5"} 6
sglang:e2e_request_latency_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="0.8"} 6
sglang:e2e_request_latency_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2",le="+Inf"} 11228
sglang:e2e_request_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 3116093.850019932
sglang:e2e_request_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 11228

# HELP sglang:gen_throughput The generate throughput (token/s)
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 86.50814177726902
sglang:gen_throughput{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 86.50814177726902

# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker1"} 8128902
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct",source="worker2"} 8128902"#;
    // result output intentionally omitted for clippy compliance
    assert_eq_sorted(&result, expected);
}

#[test]
fn test_mismatched_labels_pd_mode() {
    // Simulates PD disaggregation: prefill worker has no dp_rank label,
    // decode worker has dp_rank. Previously this caused a 500 error.
    let prefill = MetricPack {
        labels: vec![("worker_addr".to_string(), "prefill:8000".to_string())],
        metrics_text: r#"
# HELP sglang_num_running_reqs The number of running requests
# TYPE sglang_num_running_reqs gauge
sglang_num_running_reqs{engine_type="prefill",model_name="kimi",tp_rank="0"} 10
"#
        .to_string(),
    };
    let decode = MetricPack {
        labels: vec![("worker_addr".to_string(), "decode:8000".to_string())],
        metrics_text: r#"
# HELP sglang_num_running_reqs The number of running requests
# TYPE sglang_num_running_reqs gauge
sglang_num_running_reqs{dp_rank="0",engine_type="decode",model_name="kimi",tp_rank="0"} 20
"#
        .to_string(),
    };

    // Should succeed — not return an error
    let result = aggregate_metrics(vec![prefill, decode]).unwrap();

    // Both workers should appear in the output
    assert!(result.contains("prefill:8000"), "prefill worker missing");
    assert!(result.contains("decode:8000"), "decode worker missing");
    // Prefill sample should get dp_rank="" (padded)
    assert!(
        result.contains(r#"dp_rank="""#),
        "prefill should have empty dp_rank label"
    );
    // Decode sample should keep dp_rank="0"
    assert!(
        result.contains(r#"dp_rank="0""#),
        "decode should keep dp_rank=0"
    );
}

fn assert_eq_sorted(result: &str, expected: &str) {
    // Split into lines and sort to handle BTreeMap ordering issues between test environments
    let mut result_lines: Vec<_> = result.trim().lines().map(|l| l.trim()).collect();
    let mut expected_lines: Vec<_> = expected.trim().lines().map(|l| l.trim()).collect();
    result_lines.sort_unstable();
    expected_lines.sort_unstable();
    assert_eq!(result_lines, expected_lines);
}
