//! Field21 is additive: existing media-identity fields19/20 retain their wire.

use prost::Message;
use smg_grpc_client::sglang_proto as sg;

#[derive(Clone, PartialEq, Message)]
struct BeforeHandoffGenerate {
    #[prost(string, tag = "1")]
    request_id: String,
    #[prost(bool, tag = "19")]
    return_media_identity: bool,
    #[prost(message, optional, tag = "20")]
    media_identity: Option<sg::MediaIdentity>,
}

#[test]
fn old_media_identity_wire_keeps_both_fields_and_has_no_handoff() {
    let old = BeforeHandoffGenerate {
        request_id: "old-request".into(),
        return_media_identity: true,
        media_identity: Some(sg::MediaIdentity {
            version: 1,
            processor_key: "original-processor".into(),
            ..Default::default()
        }),
    };
    let current = sg::GenerateRequest::decode(old.encode_to_vec().as_slice()).unwrap();
    assert!(current.return_media_identity);
    assert_eq!(current.media_identity, old.media_identity);
    assert!(current.encode_handoff.is_none());
    let roundtrip = BeforeHandoffGenerate::decode(current.encode_to_vec().as_slice()).unwrap();
    assert_eq!(roundtrip, old);
}

#[test]
fn new_typed_handoff_roundtrips_without_overwriting_media_identity() {
    let current = sg::GenerateRequest {
        request_id: "new-request".into(),
        return_media_identity: true,
        media_identity: Some(sg::MediaIdentity {
            version: 1,
            processor_key: "existing-identity".into(),
            ..Default::default()
        }),
        encode_handoff: Some(sg::EncodeHandoff {
            request_id: "new-request".into(),
            encoder_urls: vec![
                "grpc://encoder-0:5001".into(),
                "grpc://encoder-1:5002".into(),
            ],
            num_items_assigned: vec![1, 0],
        }),
        ..Default::default()
    };
    let wire = current.encode_to_vec();
    assert_eq!(
        sg::GenerateRequest::decode(wire.as_slice()).unwrap(),
        current
    );
    let old = BeforeHandoffGenerate::decode(wire.as_slice()).unwrap();
    assert_eq!(old.request_id, current.request_id);
    assert!(old.return_media_identity);
    assert_eq!(old.media_identity, current.media_identity);
    // Older binaries ignore field21: this is compatibility, not feature support.
    let old_roundtrip = sg::GenerateRequest::decode(old.encode_to_vec().as_slice()).unwrap();
    assert!(old_roundtrip.encode_handoff.is_none());
}
