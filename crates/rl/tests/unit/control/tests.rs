use super::*;

#[test]
fn concrete_hosts_pass_through() {
    assert_eq!(
        resolve_control_url("http://10.0.0.5:40100", "grpc://10.0.0.5:30000"),
        "http://10.0.0.5:40100"
    );
    assert_eq!(
        resolve_control_url("http://ctl.internal:8001/", "grpc://a:1"),
        "http://ctl.internal:8001"
    );
}

#[test]
fn wildcard_hosts_take_the_worker_host() {
    assert_eq!(
        resolve_control_url("http://0.0.0.0:40100", "grpc://10.0.0.5:30000"),
        "http://10.0.0.5:40100"
    );
    assert_eq!(
        resolve_control_url("http://[::]:40100", "grpc://[fd00::5]:30000"),
        "http://[fd00::5]:40100"
    );
    assert_eq!(
        resolve_control_url("http://:40100", "grpc://10.0.0.5:30000"),
        "http://10.0.0.5:40100"
    );
}

#[test]
fn a_worker_without_a_host_keeps_the_label_as_advertised() {
    assert_eq!(
        resolve_control_url("http://0.0.0.0:40100", "ipc:///tmp/engine.sock"),
        "http://0.0.0.0:40100",
        "there is no worker host to substitute"
    );
}

#[test]
fn schemeless_or_odd_input_is_returned_unchanged() {
    assert_eq!(
        resolve_control_url("10.0.0.5:40100", "grpc://a:1"),
        "10.0.0.5:40100"
    );
}
