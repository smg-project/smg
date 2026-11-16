use super::super::*;

fn sampling(frames: usize) -> VideoSamplingInfo {
    VideoSamplingInfo {
        source_fps: 30.0,
        frame_indices: (0..frames).map(|i| i * 15).collect(),
    }
}

fn rgb_video(width: u32, height: u32, frames: usize) -> DecodedVideoFrames {
    let len = (width * height * 3) as usize;
    let mut data = Vec::with_capacity(len * frames);
    let mut descs = Vec::with_capacity(frames);
    for i in 0..frames {
        descs.push(DecodedRgbFrame {
            width,
            height,
            offset: i * len,
            len,
        });
        data.extend(std::iter::repeat_n(0u8, len));
    }
    DecodedVideoFrames::Rgb {
        video: DecodedRgbVideo::new(Bytes::from(data), descs),
        sample_fps: 2.0,
        sampling: Some(sampling(frames)),
    }
}

#[test]
fn sampling_metadata_survives_the_cap_unchanged() {
    // Rescaled, left alone, and uncapped: the cap only touches pixels.
    for (decoded, cap) in [
        (rgb_video(1920, 1080, 3), Some(504)),
        (rgb_video(320, 240, 3), Some(1008)),
        (rgb_video(1920, 1080, 3), None),
    ] {
        let DecodedVideoFrames::Rgb {
            sample_fps,
            sampling: capped,
            ..
        } = cap_decoded_frames(decoded, cap)
        else {
            panic!("expected RGB");
        };
        assert_eq!(sample_fps, 2.0);
        assert_eq!(capped, Some(sampling(3)));
    }
}

/// A descriptor that disagrees with its buffer must not size the
/// allocation: the capacity is clamped to the source buffer, and the
/// per-frame bounds check then hands the clip back untouched.
#[test]
fn inconsistent_frame_descriptors_do_not_drive_the_allocation() {
    let small = (100 * 100 * 3) as usize;
    let big = (1920 * 1080 * 3) as usize;
    let data = vec![0u8; small + big];
    let frames = vec![
        // Inside the cap, so its `len` feeds the capacity sum directly,
        // and that `len` is nonsense.
        DecodedRgbFrame {
            width: 100,
            height: 100,
            offset: 0,
            len: isize::MAX as usize,
        },
        // Over the cap, so the "nothing to cap" early return does not fire.
        DecodedRgbFrame {
            width: 1920,
            height: 1080,
            offset: small,
            len: big,
        },
    ];
    let decoded = DecodedVideoFrames::Rgb {
        video: DecodedRgbVideo::new(Bytes::from(data), frames),
        sample_fps: 2.0,
        sampling: None,
    };

    let DecodedVideoFrames::Rgb { video, .. } = cap_decoded_frames(decoded, Some(504)) else {
        panic!("expected RGB");
    };
    // Left to downstream validation rather than reshaped or reallocated.
    assert_eq!(video.frames[0].len, isize::MAX as usize);
    assert_eq!(
        (video.frames[1].width, video.frames[1].height),
        (1920, 1080)
    );
}

#[test]
fn capped_dimensions_fits_the_box_and_keeps_aspect() {
    // 16:9 source, long side capped to 504.
    assert_eq!(capped_dimensions(1920, 1080, 504), Some((504, 284)));
    // Portrait caps on height.
    assert_eq!(capped_dimensions(1080, 1920, 504), Some((284, 504)));
}

#[test]
fn capped_dimensions_never_upscales() {
    assert_eq!(capped_dimensions(320, 240, 1008), None);
    assert_eq!(capped_dimensions(504, 284, 504), None);
}

#[test]
fn rgb_frames_are_rescaled_and_stay_self_consistent() {
    // The raw path has no per-frame header, so every descriptor must agree
    // with the buffer it points into.
    let capped = cap_decoded_frames(rgb_video(1920, 1080, 3), Some(504));
    let DecodedVideoFrames::Rgb { video, .. } = capped else {
        panic!("expected the RGB representation to be preserved");
    };

    assert_eq!(video.frames.len(), 3);
    let mut expected_offset = 0;
    for frame in &video.frames {
        assert_eq!((frame.width, frame.height), (504, 284));
        assert_eq!(frame.len, (504 * 284 * 3) as usize);
        assert_eq!(frame.offset, expected_offset);
        assert!(video
            .data
            .get(frame.offset..frame.offset + frame.len)
            .is_some());
        expected_offset += frame.len;
    }
    assert_eq!(video.data.len(), expected_offset);
}

#[test]
fn rgb_frames_within_the_cap_are_untouched() {
    let original = rgb_video(320, 240, 2);
    let capped = cap_decoded_frames(original, Some(1008));
    let DecodedVideoFrames::Rgb { video, .. } = capped else {
        panic!("expected RGB");
    };
    assert_eq!(video.frames[0].width, 320);
    assert_eq!(video.frames[0].height, 240);
}

#[test]
fn absent_cap_leaves_frames_alone() {
    let capped = cap_decoded_frames(rgb_video(1920, 1080, 1), None);
    let DecodedVideoFrames::Rgb { video, .. } = capped else {
        panic!("expected RGB");
    };
    assert_eq!(
        (video.frames[0].width, video.frames[0].height),
        (1920, 1080)
    );
}

#[test]
fn image_frames_are_rescaled_too() {
    // The PPM/DynamicImage backend must honour the same cap.
    let decoded = DecodedVideoFrames::Images {
        frames: vec![image::DynamicImage::new_rgb8(1920, 1080)],
        sample_fps: 2.0,
        sampling: Some(sampling(1)),
    };
    let DecodedVideoFrames::Images {
        frames,
        sampling: capped,
        ..
    } = cap_decoded_frames(decoded, Some(504))
    else {
        panic!("expected images");
    };
    assert_eq!((frames[0].width(), frames[0].height()), (504, 284));
    assert_eq!(capped, Some(sampling(1)));
}
