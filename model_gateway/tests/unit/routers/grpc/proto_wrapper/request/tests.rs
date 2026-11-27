use super::*;

/// The prefill side's request says its embeddings come from an encode
/// worker while it carries encode rendezvous rooms; its pixel-free decode
/// clone, whose rooms are cleared, waits on the prefill's handoff instead,
/// and a request with no rooms never did.
#[test]
fn a_request_awaits_encode_only_while_it_carries_encode_rooms() {
    let mut request = ProtoGenerateRequest::TokenSpeed(Box::default());
    assert!(!request.awaits_encode(), "no rooms, no encode wait");
    request.set_encode_bootstrap_info(Vec::new());
    assert!(
        !request.awaits_encode(),
        "an empty room list is no encode wait"
    );
    request.set_encode_bootstrap_info(vec![EncodeItemBootstrapInfo {
        item_index: 0,
        bootstrap_host: "encode-worker".to_string(),
        bootstrap_port: 9000,
        bootstrap_room: 41,
    }]);
    assert!(
        request.awaits_encode(),
        "the prefill side waits on the encoder"
    );
    let mut decode = request.clone_without_mm_pixels();
    decode.clear_encode_bootstrap_info();
    assert!(
        !decode.awaits_encode(),
        "the decode side waits on the prefill's handoff, not on the encoder"
    );
    assert!(request.awaits_encode(), "the prefill side still does");
}
