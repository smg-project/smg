use super::*;
use crate::worker::{BasicWorkerBuilder, Worker};

pub(super) fn image_uri(url: String) -> MediaContentPart {
    MediaContentPart::ImageUrl {
        url,
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    }
}

pub(super) fn originals() -> Vec<String> {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../e2e_test/chat_completions/test_epd_multimodal.py"
    ));
    let png = |name: &str| {
        let value = source
            .split_once(&format!("{name} = (\n"))
            .unwrap()
            .1
            .split_once("\n)")
            .unwrap()
            .0;
        let encoded = value
            .lines()
            .map(|line| line.trim().trim_matches('"'))
            .collect::<String>();
        format!("data:image/png;base64,{encoded}")
    };
    let jpeg = |name: &str| {
        format!(
            "data:image/jpeg;base64,{}",
            STANDARD.encode(
                std::fs::read(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../e2e_test/fixtures/images")
                        .join(name)
                )
                .unwrap()
            )
        )
    };
    vec![
        png("RED_PNG_B64"),
        png("BLUE_PNG_B64"),
        jpeg("dog.jpg"),
        jpeg("pug.jpg"),
    ]
}

pub(super) fn encoder(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Encode)
            .runtime_type(RuntimeType::Sglang)
            .connection_mode(ConnectionMode::Grpc)
            .build(),
    )
}

#[test]
fn epd_sglang_preserves_original_media_and_maps_aba_without_reordering() {
    let originals = originals();
    let media = MediaPlan::new(originals.iter().cloned().map(image_uri));
    assert_eq!(image_urls(&media).unwrap(), originals);
    let bytes = STANDARD
        .decode(originals[2].split_once(',').unwrap().1)
        .unwrap();
    let inline = MediaPlan::new([MediaContentPart::ImageData {
        data: bytes.clone(),
        mime_type: None,
        uuid: None,
        detail: None,
    }]);
    let urls = image_urls(&inline).unwrap();
    assert_eq!(
        STANDARD.decode(urls[0].split_once(',').unwrap().1).unwrap(),
        bytes
    );
    let a = encoder("grpc://a:5001");
    let b = encoder("grpc://b:5002");
    let assignments = [a.clone(), b.clone(), a]
        .into_iter()
        .enumerate()
        .map(|(item_index, worker)| EncodeWorkerAssignment { item_index, worker })
        .collect::<Vec<_>>();
    let mapped = contiguous(&assignments).unwrap();
    assert_eq!(
        mapped.iter().map(|a| a.worker.url()).collect::<Vec<_>>(),
        ["grpc://a:5001", "grpc://a:5001", "grpc://b:5002"]
    );
    assert_eq!(
        mapped.iter().map(|a| a.item_index).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    let handoff = handoff("final-rid", &mapped).unwrap();
    assert_eq!(handoff.encoder_urls, ["grpc://a:5001", "grpc://b:5002"]);
    assert_eq!(handoff.num_items_assigned, [2, 1]);
    assert_eq!(image_urls(&media).unwrap(), originals);
    let fake = Arc::new(BasicWorkerBuilder::new("grpc://fake:5001").build()) as Arc<dyn Worker>;
    assert!(contiguous(&[EncodeWorkerAssignment {
        item_index: 0,
        worker: fake
    }])
    .is_err());
    let invalid = encoder("grpc://a:5001/");
    assert!(contiguous(&[EncodeWorkerAssignment {
        item_index: 0,
        worker: invalid
    }])
    .is_err());
}
