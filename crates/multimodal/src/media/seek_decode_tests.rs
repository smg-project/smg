use std::collections::HashSet;

use super::*;

fn cfg() -> VideoFetchConfig {
    VideoFetchConfig {
        min_frames: 4,
        max_frames: 8,
        sample_fps: 2.0,
        max_long_side_pixel: None,
        sampling: FrameSampling::Even,
    }
}

fn long_clip_metadata() -> VideoMetadata {
    VideoMetadata {
        width: 64,
        height: 48,
        duration_seconds: Some(36_000.0),
        source_fps: Some(30.0),
        total_frames: Some(1_080_000),
        container_frames: Some(1_080_000),
        from_container: false,
    }
}

/// 100 s of 64x48 at 30 fps as an mp4, a keyframe every 48 frames with
/// two B-frames between references; `false` when ffmpeg is not on PATH.
async fn generate_clip(path: &Path) -> bool {
    Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=30:duration=100",
            "-c:v",
            "mpeg4",
            "-g",
            "48",
            "-bf",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|status| status.success())
}

/// `source` remuxed to start 7 s into its container's timeline.
async fn remux_starting_late(source: &Path, path: &Path) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(source)
        .args(["-c", "copy", "-output_ts_offset", "7", "-y"])
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|status| status.success())
}

async fn have_probe_tool() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|status| status.success())
}

fn frame_bytes(video: &DecodedRgbVideo) -> Vec<(u32, u32, Vec<u8>)> {
    video
        .frame_refs()
        .expect("frame refs")
        .iter()
        .map(|frame| (frame.width, frame.height, frame.data.to_vec()))
        .collect()
}

/// Skipped when ffmpeg or ffprobe is not on PATH.
#[tokio::test]
async fn a_long_clip_sampled_sparsely_is_decoded_by_seeking_to_the_same_frames() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("clip.mp4");
    if !(generate_clip(&path).await && have_probe_tool().await) {
        return;
    }
    let input_bytes = fs::read(&path).await.expect("generated clip").len();
    let metadata = probe_video_metadata(&path).await.expect("metadata");
    assert_eq!(metadata.total_frames, Some(3000));

    let loader_cfg = VideoFetchConfig {
        sampling: FrameSampling::UpTo { max_frames: 8 },
        ..cfg()
    };
    for cfg in [cfg(), loader_cfg] {
        let selection = FrameSelection::from_metadata(metadata, cfg).expect("a selection");
        assert_eq!(selection.unique_count(), 8);
        let plan = seek_plan_for(&path, &selection)
            .await
            .expect("eight picks over 3000 frames are a clear win");
        assert_eq!(plan.segments.len(), 8);
        assert!(
            plan.decoded_frames() * 4 < 3000,
            "{} frames decoded",
            plan.decoded_frames()
        );

        // The same bytes as the sequential decode, pick for pick.
        let seeked = decode_video_with_ffmpeg_seek(&path, metadata, &selection, &plan)
            .await
            .expect("seek decode");
        let sequential = decode_video_with_ffmpeg_ppm(&path, cfg, metadata, Some(&selection))
            .await
            .expect("sequential decode");
        assert_eq!(frame_bytes(&seeked), frame_bytes(&sequential));
        assert_eq!(seeked.frames.len(), selection.frame_indices.len());
        let distinct: HashSet<Vec<u8>> = frame_bytes(&seeked)
            .into_iter()
            .map(|(_, _, data)| data)
            .collect();
        assert_eq!(distinct.len(), 8, "eight distinct source frames");

        // The route takes the plan and reports the sampling.
        match decode_video_with_ffmpeg(&path, input_bytes, cfg)
            .await
            .expect("decode")
        {
            DecodedVideoFrames::Rgb {
                video, sampling, ..
            } => {
                assert_eq!(frame_bytes(&video), frame_bytes(&sequential));
                assert_eq!(sampling, Some(selection.sampling_info()));
            }
            DecodedVideoFrames::Images { .. } => {
                panic!("the headed PNG runner is not the route for this clip")
            }
        }

        // A seek that lands past its picks leaves them missing: an error
        // the sequential decode answers, never another frame.
        let mut misplaced = plan.clone();
        misplaced.segments[1].seek_seconds = plan.segments[2].seek_seconds.clone();
        let error = decode_video_with_ffmpeg_seek(&path, metadata, &selection, &misplaced)
            .await
            .expect_err("a missing pick");
        assert!(!is_video_command_timeout(&error), "{error}");
        assert!(
            error
                .to_string()
                .contains("produced 0 of its 1 picks: the seek did not land"),
            "{error}"
        );
    }

    // A container that starts 7 s into its timeline: the seeks are
    // measured from its start, the picks named by their own timestamps.
    let late = dir.path().join("late.mp4");
    assert!(remux_starting_late(&path, &late).await);
    let metadata = probe_video_metadata(&late).await.expect("metadata");
    let selection = FrameSelection::from_metadata(metadata, cfg()).expect("a selection");
    let plan = seek_plan_for(&late, &selection).await.expect("a plan");
    assert!(plan
        .segments
        .iter()
        .all(|segment| !segment.seek_seconds.starts_with('-')));
    let seeked = decode_video_with_ffmpeg_seek(&late, metadata, &selection, &plan)
        .await
        .expect("seek decode");
    let sequential = decode_video_with_ffmpeg_ppm(&late, cfg(), metadata, Some(&selection))
        .await
        .expect("sequential decode");
    assert_eq!(frame_bytes(&seeked), frame_bytes(&sequential));
}

/// Skipped when ffmpeg or ffprobe is not on PATH.
#[tokio::test]
async fn a_short_clip_keeps_the_sequential_decode() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("clip.mp4");
    if !(generate_clip(&path).await && have_probe_tool().await) {
        return;
    }
    let metadata = probe_video_metadata(&path).await.expect("metadata");
    // 3000 frames for 200 picks: fifteen frames a pick, no plan is looked into.
    let selection =
        FrameSelection::from_metadata(metadata, VideoFetchConfig::default()).expect("a selection");
    assert_eq!(selection.unique_count(), 200);
    assert!(seek_plan_for(&path, &selection).await.is_none());
}

#[test]
fn a_seek_decode_enters_the_stream_at_the_position_and_keeps_its_timestamps() {
    let args: Vec<String> = ffmpeg_seek_decode_command(1, "1791.666667")
        .as_std()
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        args,
        [
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-ss",
            "1791.666667",
            "-copyts",
            "-i"
        ]
    );
}

/// A run is bounded at its last pick: the trim before the `select` ends
/// the filter graph, and the run, as soon as the decoder is past it, so a
/// pick the seek missed costs the run its segment (the keyframe to the
/// last pick) and not the rest of the stream.
#[test]
fn a_seek_run_ends_right_after_its_last_pick() {
    let segment = SeekSegment {
        seek_seconds: "1.600000".to_string(),
        pts: vec![24_576, 25_088],
        decoded_frames: 2,
    };
    let command = ffmpeg_seek_run_command(
        Path::new("clip.mp4"),
        &segment,
        18_560,
        FPS_MODE_PASSTHROUGH,
    );
    let args: Vec<String> = command
        .as_std()
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let input = args.iter().position(|arg| arg == "-i").expect("an input");
    assert_eq!(
        args[input + 1..].to_vec(),
        [
            "clip.mp4",
            "-vf",
            r"trim=end_pts=25089,select='eq(pts\,24576)+eq(pts\,25088)',format=rgba,format=rgb24",
            "-frames:v",
            "2",
            "-fps_mode",
            "passthrough",
            "-fs",
            "18560",
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "-pix_fmt",
            "rgb24",
            "pipe:1"
        ]
    );
}

/// The cap is in pixels, so it follows the clip's size: ten hours at
/// 30 fps are 1,080,000 source frames, 3.3 gigapixels at 64x48 and
/// 2.2 terapixels at 1920x1080, over the terapixel.
#[test]
fn a_decode_past_the_work_cap_is_refused_before_it_starts() {
    let metadata = long_clip_metadata();
    let selection =
        FrameSelection::from_metadata(metadata, VideoFetchConfig::default()).expect("a selection");
    assert_eq!(selection.frame_indices.len(), 768);
    assert_eq!(selection.sequential_frames(), 1_080_000);
    assert!(ensure_decode_work_limit(selection.sequential_frames(), &selection, metadata).is_ok());
    let full_hd = VideoMetadata {
        width: 1920,
        height: 1080,
        ..metadata
    };
    let error = ensure_decode_work_limit(selection.sequential_frames(), &selection, full_hd)
        .expect_err("ten hours of 1920x1080 at 30 fps are over the cap");
    assert!(
        is_video_budget_error(&error),
        "the refusal ends the decode; no fallback runner or resampling answers it"
    );
    let text = error.to_string();
    assert!(
        text.contains(
            "decoding 1080000 of its 1080000 source frames of 1920x1080, 2239488000000 pixels"
        ) && text.contains("SMG_VIDEO_MAX_DECODE_PIXELS=1000000000000")
            && text.contains("a clip of 36000 s"),
        "{text}"
    );
    // The frames one cap admits scale with the frame: four times as many
    // at 640x360 as at 1280x720.
    let hd = VideoMetadata {
        width: 1280,
        height: 720,
        ..metadata
    };
    assert!(ensure_decode_work_limit(1_085_069, &selection, hd).is_ok());
    assert!(ensure_decode_work_limit(1_085_070, &selection, hd).is_err());
    let sd = VideoMetadata {
        width: 640,
        height: 360,
        ..metadata
    };
    assert!(ensure_decode_work_limit(4_340_277, &selection, sd).is_ok());
    assert!(ensure_decode_work_limit(4_340_278, &selection, sd).is_err());
}

/// A container that overstates the stream's size is judged again on the
/// probe's size before its refusal stands: the decoder's size fits the
/// cap, so the decode goes on with it. A probe that states no size or
/// cannot run leaves the refusal standing, a size the probe itself stated
/// is not probed again, and a size under the cap asks no probe.
#[tokio::test]
async fn an_overstated_container_size_is_judged_again_on_the_probes_size() {
    let metadata = long_clip_metadata();
    let selection =
        FrameSelection::from_metadata(metadata, VideoFetchConfig::default()).expect("a selection");
    let overstated = VideoMetadata {
        width: 65_535,
        height: 65_535,
        from_container: true,
        ..metadata
    };
    let frames = 1_000;
    assert!(ensure_decode_work_limit(frames, &selection, overstated).is_err());
    let path = Path::new("overstated.mp4");
    let decoders = ProbedVideoInfo {
        width: Some(64),
        height: Some(48),
        duration_seconds: metadata.duration_seconds,
        source_fps: metadata.source_fps,
        total_frames: metadata.total_frames,
        container_frames: metadata.container_frames,
    };
    probe_switch::ANSWER.with(|answer| answer.set(Some(decoders)));
    let judged = decode_work_limit_on_the_decoders_size(frames, &selection, overstated, path)
        .await
        .expect("the decoder's size fits the cap");
    assert_eq!((judged.width, judged.height), (64, 48));
    assert!(!judged.from_container);
    // A probe that states no size leaves the refusal standing.
    probe_switch::ANSWER.with(|answer| {
        answer.set(Some(ProbedVideoInfo {
            width: Some(0),
            height: Some(0),
            ..decoders
        }));
    });
    let error = decode_work_limit_on_the_decoders_size(frames, &selection, overstated, path)
        .await
        .expect_err("no size from the probe: the container's refusal stands");
    assert!(is_video_budget_error(&error));
    probe_switch::ANSWER.with(|answer| answer.set(None));
    // So does a probe that cannot run.
    probe_switch::UNAVAILABLE.with(|off| off.set(true));
    let error = decode_work_limit_on_the_decoders_size(frames, &selection, overstated, path)
        .await
        .expect_err("no probe: the container's refusal stands");
    probe_switch::UNAVAILABLE.with(|off| off.set(false));
    assert!(is_video_budget_error(&error));
    // A size the probe itself stated is final: no second probe is asked.
    probe_switch::ANSWER.with(|answer| answer.set(Some(decoders)));
    let probed = VideoMetadata {
        from_container: false,
        ..overstated
    };
    assert!(
        decode_work_limit_on_the_decoders_size(frames, &selection, probed, path)
            .await
            .is_err()
    );
    probe_switch::ANSWER.with(|answer| answer.set(None));
    // A size under the cap goes on as it is, no probe asked.
    probe_switch::UNAVAILABLE.with(|off| off.set(true));
    let kept = decode_work_limit_on_the_decoders_size(frames, &selection, metadata, path)
        .await
        .expect("under the cap");
    probe_switch::UNAVAILABLE.with(|off| off.set(false));
    assert_eq!((kept.width, kept.height), (metadata.width, metadata.height));
}

#[test]
fn a_run_past_the_process_budget_is_reported_in_the_clips_terms() {
    let metadata = long_clip_metadata();
    let selection =
        FrameSelection::from_metadata(metadata, VideoFetchConfig::default()).expect("a selection");
    let error = decode_budget_exceeded(
        MediaConnectorError::VideoDecode("ffmpeg timed out after 30.000 seconds".to_string()),
        &selection,
        metadata,
        54_000,
    );
    assert!(is_video_command_timeout(&error));
    let text = error.to_string();
    assert!(
        text.contains("decoding 54000 source frames of a clip of 36000 s to sample 768")
            && text.contains("SMG_VIDEO_PROCESS_TIMEOUT_SECS"),
        "{text}"
    );
    // Any other error is left as it is.
    let other = MediaConnectorError::PayloadTooLarge {
        media: "video",
        limit: 1,
    };
    assert!(matches!(
        decode_budget_exceeded(other, &selection, metadata, 1),
        MediaConnectorError::PayloadTooLarge { .. }
    ));
}
