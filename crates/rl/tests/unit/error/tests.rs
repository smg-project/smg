use super::*;

#[test]
fn codes_and_statuses() {
    let e = RlError::WorkerNotFound("abc".into());
    assert_eq!(e.status(), StatusCode::NOT_FOUND);
    assert_eq!(e.to_json()["error"], "worker_not_found");
    assert_eq!(e.to_json()["id"], "abc");

    let e = RlError::InvalidSelector {
        offset: 7,
        message: "expected operator".into(),
    };
    assert_eq!(e.status(), StatusCode::BAD_REQUEST);
    assert_eq!(e.to_json()["offset"], 7);

    let e = RlError::UpstreamTimeout {
        worker_id: "w".into(),
        url: "http://x".into(),
        timeout_secs: 5,
    };
    assert_eq!(e.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(e.to_json()["url"], "http://x");

    let e = RlError::NoControlEndpoint {
        worker_id: "g".into(),
        url: "grpc://x:1".into(),
        connection_mode: "grpc".into(),
        message: "no `rl.control_url` label".into(),
    };
    assert_eq!(e.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(e.to_json()["error"], "no_control_endpoint");
    assert_eq!(e.to_json()["connection_mode"], "grpc");
    assert!(e.to_string().contains("no `rl.control_url` label"));
}
