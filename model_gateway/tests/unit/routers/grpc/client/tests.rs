use super::*;

/// The predicate and the connect name the same runtimes. A connect
/// with a wire dials, and nothing listens here. A connect with none
/// refuses the name ahead of the dial.
#[tokio::test]
async fn the_client_has_a_wire_for_each_runtime_its_connect_takes() {
    let all = [
        RuntimeType::Unspecified,
        RuntimeType::Sglang,
        RuntimeType::Vllm,
        RuntimeType::Trtllm,
        RuntimeType::Mlx,
        RuntimeType::TokenSpeed,
        RuntimeType::Generic,
        RuntimeType::External,
    ];
    for runtime in all {
        let refused = GrpcClient::connect("grpc://127.0.0.1:1", &runtime.to_string()).await;
        let refused = refused.err().map(|error| error.to_string());
        let refused = refused.expect("nothing listens");
        let takes_the_name = !refused.starts_with("Unknown runtime type");
        assert_eq!(
            GrpcClient::has_wire(runtime),
            takes_the_name,
            "{runtime}: {refused}"
        );
    }
}
