use super::*;

fn sample() -> RlWorkerInfo {
    RlWorkerInfo {
        id: "w".to_string(),
        url: "http://a:1".to_string(),
        base_url: "http://a:1".to_string(),
        api_key: Some("hunter2-secret".to_string()),
        model_id: "m".to_string(),
        runtime: RuntimeType::Sglang,
        worker_type: WorkerType::Regular,
        connection_mode: ConnectionMode::Http,
        status: WorkerStatus::Ready,
        is_dp_aware: false,
        dp_size: None,
        labels: HashMap::new(),
        control_url: None,
        control_client: Err("no client".to_string()),
        control_calls: None,
    }
}

#[test]
fn debug_output_redacts_the_api_key() {
    let info = sample();
    let dbg = format!("{info:?}");
    assert!(!dbg.contains("hunter2-secret"), "{dbg}");
    assert!(dbg.contains("api_key: Some(\"<redacted>\")"), "{dbg}");
    assert!(dbg.contains("id: \"w\""), "{dbg}");
}

#[test]
fn debug_output_shows_the_control_url() {
    let mut info = sample();
    info.control_url = Some("http://ctl:1".to_string());
    assert!(format!("{info:?}").contains("control_url: Some(\"http://ctl:1\")"));
}
