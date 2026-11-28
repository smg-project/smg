use super::*;

#[test]
fn encoder_items_to_worker_items_forwards_image_detail_and_rejects_invalid() {
    let items = encoder_items_to_worker_items(
        EncoderModality::Image,
        vec![
            EncoderMediaItem {
                url: "https://example.test/a.png".into(),
                detail: Some("low".into()),
            },
            EncoderMediaItem {
                url: "https://example.test/b.png".into(),
                detail: Some("HIGH".into()),
            },
            EncoderMediaItem {
                url: "https://example.test/c.png".into(),
                detail: Some("auto".into()),
            },
            EncoderMediaItem {
                url: "https://example.test/d.png".into(),
                detail: None,
            },
        ],
    )
    .unwrap();
    assert_eq!(items.len(), 4);
    assert_eq!(items[0].detail, Some(llm_multimodal::ImageDetail::Low));
    assert_eq!(items[1].detail, Some(llm_multimodal::ImageDetail::High));
    assert_eq!(items[2].detail, Some(llm_multimodal::ImageDetail::Auto));
    assert_eq!(items[3].detail, None);

    let err = encoder_items_to_worker_items(
        EncoderModality::Image,
        vec![EncoderMediaItem {
            url: "https://example.test/bad.png".into(),
            detail: Some("ultra".into()),
        }],
    )
    .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}
