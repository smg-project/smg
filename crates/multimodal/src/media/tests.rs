use std::io::Write as _;
/// Where the tools are installed: clips written by the encoder at several
/// frame rates and lengths, read through the container tables, equal the
/// probe tool's own report field by field (the numbers the frame
/// selection and the clip's sample rate are built from).
#[test]
fn container_tables_agree_with_the_probe_tool() {
    use std::process::Command;
    let have = |tool: &str| {
        Command::new(tool)
            .arg("-version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    };
    if !have("ffmpeg") || !have("ffprobe") {
        // Without the tools there is nothing to agree with; the unit tests
        // on synthetic containers still run.
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let cases = [
        ("24", "1", "mp4"),
        ("30000/1001", "2", "mp4"),
        ("25", "3", "mov"),
        ("60", "1", "mp4"),
        ("24000/1001", "2", "mp4"),
    ];
    for (rate, seconds, container) in cases {
        let path = dir.path().join(format!(
            "clip_{}_{seconds}.{container}",
            rate.replace('/', "_")
        ));
        // A codec every ffmpeg build carries; the tables under test are
        // the container's, whichever codec fills them.
        let encoded = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
            ])
            .arg(format!("testsrc=size=64x48:rate={rate}:duration={seconds}"))
            .args(["-pix_fmt", "yuv420p", "-c:v", "mpeg4"])
            .arg(&path)
            .output()
            .expect("ffmpeg runs");
        assert!(
            encoded.status.success(),
            "ffmpeg wrote {path:?}: {}",
            String::from_utf8_lossy(&encoded.stderr)
        );
        let container_info = crate::mp4_metadata::read(&path)
            .expect("readable")
            .unwrap_or_else(|| panic!("{path:?} not accepted by the container reader"));
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration,duration_ts,time_base,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .expect("ffprobe runs");
        assert!(output.status.success());
        let probed = parse_ffprobe_video_info(&output.stdout).expect("probe parses");
        assert_eq!(
            super::probed_from_container(&container_info),
            probed,
            "{rate} fps, {seconds} s, {container}"
        );
    }
}

/// The engine's frame budget replaces only the loader-style budget: the
/// rate-based samplers keep their rule, `0` means every frame, `None`
/// changes nothing.
#[test]
fn frame_budget_applies_to_the_loader_style_sampler_only() {
    let loader = FrameSampling::UpTo { max_frames: 32 };
    assert_eq!(
        loader.with_frame_budget(Some(16)),
        FrameSampling::UpTo { max_frames: 16 }
    );
    assert_eq!(
        loader.with_frame_budget(Some(0)),
        FrameSampling::UpTo {
            max_frames: usize::MAX
        }
    );
    assert_eq!(loader.with_frame_budget(None), loader);
    assert_eq!(
        FrameSampling::Even.with_frame_budget(Some(16)),
        FrameSampling::Even
    );
    assert_eq!(
        FrameSampling::Interval.with_frame_budget(Some(16)),
        FrameSampling::Interval
    );
    // A 50-frame clip under a budget of 16 is sampled like the loader does.
    let cfg = VideoFetchConfig {
        sampling: loader.with_frame_budget(Some(16)),
        ..VideoFetchConfig::default()
    };
    assert_eq!(super::sampled_frame_indices(50, 10.0, cfg).len(), 16);
    let cfg = VideoFetchConfig {
        sampling: loader.with_frame_budget(Some(0)),
        ..VideoFetchConfig::default()
    };
    assert_eq!(super::sampled_frame_indices(50, 10.0, cfg).len(), 50);
}

use bytes::Bytes;
use futures::stream;

use super::{
    checked_payload_length, collect_http_body_with_limit, decode_base64_with_limit,
    decoder_unavailable, effective_sample_fps, ensure_input_byte_limit,
    expected_sampled_frame_count, fallback_failed, fps_filter_for_metadata,
    parse_ffmpeg_duration_seconds, parse_ffprobe_video_info, parse_ppm_stream,
    read_file_with_limit, run_video_command_output, split_png_stream, video_metadata_from,
    video_temp_suffix, write_temp_video_file, FrameSampling, MediaConnector, MediaConnectorConfig,
    MediaConnectorError, MediaSource, ProbedVideoInfo, VideoFetchConfig, VideoMetadata,
};
use crate::error::MediaFault;

const TINY_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 4, 0,
    0, 0, 181, 28, 12, 2, 0, 0, 0, 11, 73, 68, 65, 84, 120, 218, 99, 96, 96, 0, 0, 0, 3, 0, 1, 43,
    9, 141, 84, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

#[test]
fn splits_concatenated_png_stream() {
    let mut stream = Vec::new();
    stream.extend_from_slice(TINY_PNG);
    stream.extend_from_slice(TINY_PNG);

    let frames = match split_png_stream(&stream) {
        Ok(frames) => frames,
        Err(err) => panic!("split png stream failed: {err}"),
    };
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0], TINY_PNG);
    assert_eq!(frames[1], TINY_PNG);
}

#[test]
fn parses_ffmpeg_duration() {
    let stderr = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'video.mp4':\n  Duration: 00:01:23.45, start: 0.000000, bitrate: 123 kb/s";
    assert_eq!(parse_ffmpeg_duration_seconds(stderr), Some(83.45));
}

/// A probe that states no size (ffprobe's `0x0` for a stream whose codec
/// parameters it could not find) is kept as stated, unsized, with the
/// frames and duration it did state; a sized probe is not unsized.
#[test]
fn ffprobe_metadata_keeps_a_zero_size_as_unsized() {
    let probed = |width, height| ProbedVideoInfo {
        width: Some(width),
        height: Some(height),
        duration_seconds: Some(1.0),
        source_fps: Some(30.0),
        total_frames: Some(30),
        container_frames: Some(30),
    };
    for (width, height) in [(0, 48), (64, 0), (0, 0)] {
        let metadata =
            video_metadata_from(probed(width, height), false).expect("a zero size is kept");
        assert!(metadata.is_unsized(), "{width}x{height}");
        assert_eq!((metadata.width, metadata.height), (width, height));
        assert_eq!(metadata.total_frames, Some(30));
        assert_eq!(metadata.duration_seconds, Some(1.0));
    }
    let sized = video_metadata_from(probed(64, 48), false).expect("a sized probe");
    assert!(!sized.is_unsized());
}

#[test]
fn ffprobe_metadata_prefers_short_video_stream_over_long_container() {
    let output = br#"{
        "streams": [{
            "width": 320,
            "height": 240,
            "duration": "1.000000",
            "duration_ts": 30,
            "time_base": "1/30"
        }],
        "format": {"duration": "120.000000"}
    }"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    assert_eq!(info.duration_seconds, Some(1.0));

    let cfg = VideoFetchConfig {
        min_frames: 4,
        max_frames: 8,
        sample_fps: 2.0,
        max_long_side_pixel: None,
        sampling: FrameSampling::Even,
    };
    let metadata = VideoMetadata {
        width: info.width.expect("video width"),
        height: info.height.expect("video height"),
        duration_seconds: info.duration_seconds,
        source_fps: info.source_fps,
        total_frames: info.total_frames,
        container_frames: info.container_frames,
        from_container: false,
    };
    assert_eq!(expected_sampled_frame_count(metadata, cfg), 4);
    assert_eq!(fps_filter_for_metadata(metadata, cfg), "fps=4.000000");
}

#[test]
fn ffprobe_metadata_uses_stream_time_base_before_container_duration() {
    let output = br#"{
        "streams": [{
            "width": "640",
            "height": "360",
            "duration": "N/A",
            "duration_ts": 45,
            "time_base": "1/30"
        }],
        "format": {"duration": "90.000000"}
    }"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    assert_eq!(info.width, Some(640));
    assert_eq!(info.height, Some(360));
    assert_eq!(info.duration_seconds, Some(1.5));
}

#[test]
fn detects_video_temp_suffix_from_container_header() {
    let mut mp4 = vec![0; 12];
    mp4[4..8].copy_from_slice(b"ftyp");
    assert_eq!(video_temp_suffix(&mp4), ".mp4");
    assert_eq!(video_temp_suffix(&[0x1a, 0x45, 0xdf, 0xa3]), ".webm");
    assert_eq!(video_temp_suffix(b"RIFF....AVI "), ".avi");
    assert_eq!(video_temp_suffix(b"OggS"), ".ogv");
    assert_eq!(video_temp_suffix(&[0x00, 0x00, 0x01, 0xba]), ".mpg");
    assert_eq!(video_temp_suffix(b"unknown"), ".video");
}

/// The scratch file a decoder reads from is this process's own: in a
/// directory that is not there (or not writable) the error is this
/// process's fault, whatever bytes the request sent; the same bytes in a
/// writable directory are written as they are.
#[test]
fn a_scratch_file_that_cannot_be_written_is_our_fault() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let bytes = b"not a video, never decoded";
    let written = write_temp_video_file(dir.path(), bytes).expect("a writable dir");
    assert_eq!(
        std::fs::read(written.path()).expect("the scratch file"),
        bytes
    );
    let missing = dir.path().join("missing");
    let error = write_temp_video_file(&missing, bytes).expect_err("a dir that is not there");
    assert!(matches!(error, MediaConnectorError::DecodeIo(_)), "{error}");
    assert_eq!(error.fault(), MediaFault::Internal);
}

/// A decode whose fallback failed too is judged by the fallback: on this
/// process's own I/O it stays this process's fault, with both decoders'
/// words kept; on the media it is the media's; a fallback that could not
/// start judged nothing, so the first decoder's verdict on the clip
/// stands and a corrupt clip stays the request's fault on a worker
/// without the fallback decoder.
#[test]
fn a_failed_fallback_keeps_an_internal_origin() {
    let message = "first decoder failed: bad frame; fallback failed: no space left on device";
    let bad_frame = || MediaConnectorError::VideoDecode("bad frame".to_string());
    let ours = fallback_failed(
        bad_frame(),
        MediaConnectorError::DecodeIo(std::io::Error::other("no space left on device")),
        message.to_string(),
    );
    assert_eq!(ours.fault(), MediaFault::Internal);
    assert!(ours.to_string().contains(message), "{ours}");
    let theirs = fallback_failed(
        MediaConnectorError::DecodeIo(std::io::Error::other("broken pipe")),
        MediaConnectorError::VideoDecode("no video stream".to_string()),
        message.to_string(),
    );
    assert_eq!(theirs.fault(), MediaFault::Client);
    assert!(
        matches!(theirs, MediaConnectorError::VideoDecode(_)),
        "{theirs}"
    );
    let judged_by_first = fallback_failed(
        bad_frame(),
        decoder_unavailable("ffmpeg executable not found".to_string()),
        message.to_string(),
    );
    assert_eq!(judged_by_first.fault(), MediaFault::Client);
    assert!(
        matches!(judged_by_first, MediaConnectorError::VideoDecode(_)),
        "{judged_by_first}"
    );
}

/// A decoder this process cannot start is this process's fault, not the
/// media's, whatever clip the request sent: the executable is not there,
/// or is there but cannot be run; so is a backend this build does not
/// have. The operator's words stay on the error.
#[tokio::test]
async fn a_decoder_that_cannot_start_is_our_fault() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let missing = dir.path().join("ffprobe");
    let error = run_video_command_output(tokio::process::Command::new(&missing), "ffprobe")
        .await
        .expect_err("a decoder that is not there");
    assert!(
        matches!(&error, MediaConnectorError::DecoderUnavailable(io) if io.kind() == std::io::ErrorKind::NotFound),
        "{error}"
    );
    assert_eq!(error.fault(), MediaFault::Internal);
    assert!(
        error
            .to_string()
            .contains("ffprobe executable not found; install ffprobe"),
        "{error}"
    );

    let not_executable = dir.path().join("ffmpeg");
    std::fs::write(&not_executable, b"not a program").expect("a plain file");
    let error = run_video_command_output(tokio::process::Command::new(&not_executable), "ffmpeg")
        .await
        .expect_err("a decoder that cannot be run");
    assert!(
        matches!(&error, MediaConnectorError::DecoderUnavailable(io) if io.kind() == std::io::ErrorKind::PermissionDenied),
        "{error}"
    );
    assert_eq!(error.fault(), MediaFault::Internal);
    assert!(
        error.to_string().contains("ffmpeg could not be started"),
        "{error}"
    );

    let backend = decoder_unavailable(
        "unsupported SMG_VIDEO_DECODE_BACKEND=gstreamer; expected auto, opencv, or ffmpeg"
            .to_string(),
    );
    assert_eq!(backend.fault(), MediaFault::Internal);
    assert!(
        matches!(&backend, MediaConnectorError::DecoderUnavailable(_)),
        "{backend}"
    );
    assert!(backend.to_string().contains("gstreamer"), "{backend}");
}

#[test]
fn parses_concatenated_ppm_stream() {
    let stream =
        b"P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06P6\n# comment\n1 2\n255\n\x07\x08\x09\x0a\x0b\x0c";

    let frames = match parse_ppm_stream(stream) {
        Ok(frames) => frames,
        Err(err) => panic!("parse ppm stream failed: {err}"),
    };
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].width(), 2);
    assert_eq!(frames[0].height(), 1);
    assert_eq!(frames[1].width(), 1);
    assert_eq!(frames[1].height(), 2);
}

#[test]
fn effective_sample_fps_tracks_min_and_max_frame_clamps() {
    let cfg = VideoFetchConfig {
        min_frames: 4,
        max_frames: 8,
        sample_fps: 2.0,
        max_long_side_pixel: None,
        sampling: FrameSampling::Even,
    };

    assert_eq!(effective_sample_fps(Some(1.0), cfg), 4.0);
    assert!((effective_sample_fps(Some(10.0), cfg) - 0.8).abs() < 1e-6);
    assert_eq!(effective_sample_fps(Some(3.0), cfg), 2.0);
    assert_eq!(effective_sample_fps(None, cfg), 2.0);
}

#[test]
fn rejects_truncated_ppm_stream() {
    assert!(parse_ppm_stream(b"P6\n2 1\n255\n\x01\x02").is_err());
}

#[test]
fn rejects_invalid_ppm_header() {
    assert!(parse_ppm_stream(b"P3\n1 1\n255\n\x01\x02\x03").is_err());
    assert!(parse_ppm_stream(b"P6\n1 1\n65535\n\x01\x02\x03").is_err());
}

#[test]
fn rejects_zero_dimension_ppm_stream() {
    assert!(parse_ppm_stream(b"P6\n0 1\n255\n").is_err());
    assert!(parse_ppm_stream(b"P6\n1 0\n255\n").is_err());
}

#[test]
fn rejects_overflowing_ppm_frame_size() {
    assert!(parse_ppm_stream(b"P6\n4294967295 4294967295\n255\n").is_err());
}

#[test]
fn enforces_media_payload_limit_for_each_label() {
    for media in ["image", "video", "audio"] {
        assert!(ensure_input_byte_limit(4, 4, media).is_ok());
        assert!(matches!(
            ensure_input_byte_limit(5, 4, media),
            Err(MediaConnectorError::PayloadTooLarge { media: actual, limit: 4 })
                if actual == media
        ));
    }

    assert!(checked_payload_length(usize::MAX, 1, usize::MAX, "audio").is_err());
}

#[test]
fn enforces_base64_payload_limit_before_and_after_decode() {
    for media in ["image", "video", "audio"] {
        assert_eq!(decode_base64_with_limit("AAAA", 3, media).unwrap().len(), 3);
        assert!(matches!(
            decode_base64_with_limit("AAAA", 2, media),
            Err(MediaConnectorError::PayloadTooLarge { media: actual, limit: 2 })
                if actual == media
        ));
    }

    assert!(matches!(
        decode_base64_with_limit("AAAAAAAAAAAAAAAA", 8, "audio"),
        Err(MediaConnectorError::PayloadTooLarge {
            media: "audio",
            limit: 8
        })
    ));
    assert!(matches!(
        decode_base64_with_limit("AAAAAAAAAAAA", 8, "audio"),
        Err(MediaConnectorError::PayloadTooLarge {
            media: "audio",
            limit: 8
        })
    ));
}

#[tokio::test]
async fn reads_files_at_limit_and_rejects_limit_plus_one() -> Result<(), Box<dyn std::error::Error>>
{
    let mut file = tempfile::NamedTempFile::new()?;
    file.write_all(b"12345")?;
    file.flush()?;

    for media in ["image", "video", "audio"] {
        let bytes = read_file_with_limit(file.path(), 5, media).await?;
        assert_eq!(bytes, Bytes::from_static(b"12345"));

        assert!(matches!(
            read_file_with_limit(file.path(), 4, media).await,
            Err(MediaConnectorError::PayloadTooLarge { media: actual, limit: 4 })
                if actual == media
        ));
    }
    Ok(())
}

#[tokio::test]
async fn collects_http_body_with_content_and_streaming_limits() -> Result<(), MediaConnectorError> {
    let known_oversized = reqwest::Response::from(http::Response::new(reqwest::Body::from(
        Bytes::from_static(b"12345"),
    )));
    assert!(matches!(
        collect_http_body_with_limit(known_oversized, 4, "image").await,
        Err(MediaConnectorError::PayloadTooLarge {
            media: "image",
            limit: 4
        })
    ));

    let oversized_stream = stream::iter([
        Ok::<_, std::io::Error>(Bytes::from_static(b"123")),
        Ok(Bytes::from_static(b"45")),
    ]);
    let unknown_oversized = reqwest::Response::from(http::Response::new(
        reqwest::Body::wrap_stream(oversized_stream),
    ));
    assert!(matches!(
        collect_http_body_with_limit(unknown_oversized, 4, "video").await,
        Err(MediaConnectorError::PayloadTooLarge {
            media: "video",
            limit: 4
        })
    ));

    let within_limit_stream = stream::iter([
        Ok::<_, std::io::Error>(Bytes::from_static(b"12")),
        Ok(Bytes::from_static(b"34")),
    ]);
    let within_limit = reqwest::Response::from(http::Response::new(reqwest::Body::wrap_stream(
        within_limit_stream,
    )));
    let body = collect_http_body_with_limit(within_limit, 4, "audio").await?;
    assert_eq!(body, Bytes::from_static(b"1234"));
    Ok(())
}

#[test]
fn sampled_frame_indices_preserve_min_frames_for_short_clips() {
    let cfg = VideoFetchConfig {
        min_frames: 4,
        max_frames: 8,
        sample_fps: 2.0,
        max_long_side_pixel: None,
        sampling: FrameSampling::Even,
    };
    let indices = super::sampled_frame_indices(1, 30.0, cfg);
    assert_eq!(indices, vec![0, 0, 0, 0]);
    assert_eq!(super::counted_frame_indices(&indices), vec![(0, 4)]);
}

#[test]
fn temporal_group_sampling_picks_the_reference_processors_frames() {
    // The frames the Muse Spark family's reference video processor picks
    // at its 2 fps rate (`torch.linspace(0, n - 1, count).long()` for
    // `count = min(trunc(n x 2 / fps), cap, n)` aligned down to pairs),
    // recorded from torch for these clips.
    let cfg = VideoFetchConfig {
        sampling: FrameSampling::TemporalGroups {
            frame_factor: 2,
            max_frames: 600,
        },
        ..VideoFetchConfig::default()
    };
    assert_eq!(
        super::sampled_frame_indices(17, 8.0, cfg),
        vec![0, 5, 10, 16]
    );
    // Shorter than one pair: every frame; one frame alone.
    assert_eq!(super::sampled_frame_indices(1, 30.0, cfg), vec![0]);
    assert_eq!(super::sampled_frame_indices(3, 30.0, cfg), vec![0, 2]);
    assert_eq!(super::sampled_frame_indices(45, 30.0, cfg), vec![0, 44]);
    assert_eq!(super::sampled_frame_indices(11, 8.0, cfg), vec![0, 10]);
    assert_eq!(super::sampled_frame_indices(8, 8.0, cfg), vec![0, 7]);
    let long = super::sampled_frame_indices(300, 8.0, cfg);
    assert_eq!(long.len(), 74);
    assert_eq!(&long[..6], &[0, 4, 8, 12, 16, 20]);
    assert_eq!(&long[35..39], &[143, 147, 151, 155]);
    assert_eq!(&long[70..], &[286, 290, 294, 299]);
    let ten_minutes = super::sampled_frame_indices(18000, 30.0, cfg);
    assert_eq!(ten_minutes.len(), 600);
    assert_eq!(&ten_minutes[..6], &[0, 30, 60, 90, 120, 150]);
    assert_eq!(&ten_minutes[298..302], &[8954, 8984, 9014, 9044]);
    assert_eq!(&ten_minutes[596..], &[17908, 17938, 17968, 17999]);
    let ntsc = super::sampled_frame_indices(301, 29.970_029_970_029_97, cfg);
    assert_eq!(ntsc.len(), 20);
    assert_eq!(&ntsc[..6], &[0, 15, 31, 47, 63, 78]);
    assert_eq!(&ntsc[8..12], &[126, 142, 157, 173]);
    assert_eq!(&ntsc[16..], &[252, 268, 284, 300]);
    // The reference classes' own 96-frame budget.
    let class_default = VideoFetchConfig {
        sampling: FrameSampling::TemporalGroups {
            frame_factor: 2,
            max_frames: 96,
        },
        ..VideoFetchConfig::default()
    };
    let capped = super::sampled_frame_indices(18000, 30.0, class_default);
    assert_eq!(capped.len(), 96);
    assert_eq!(&capped[..4], &[0, 189, 378, 568]);
    assert_eq!(&capped[93..], &[17620, 17809, 17999]);
    // An unknown rate is taken as 24 fps, as the reference assumes.
    assert_eq!(super::sampled_frame_indices(48, 0.0, cfg).len(), 4);
}

#[test]
fn even_rounded_sampling_picks_the_reference_processors_frames() {
    // The frames the Qwen3-VL family's reference video processor picks at
    // its 2 fps rate, as the engine's own loader recorded them for these
    // clips: trunc(duration x 2) frames, at least 4, never more than the
    // clip has, at the nearest frames of an even spread (ties to even).
    let cfg = VideoFetchConfig {
        sampling: FrameSampling::EvenRounded,
        ..VideoFetchConfig::default()
    };
    assert_eq!(
        super::sampled_frame_indices(24, 24.0, cfg),
        vec![0, 8, 15, 23]
    );
    assert_eq!(
        super::sampled_frame_indices(120, 24.0, cfg),
        vec![0, 13, 26, 40, 53, 66, 79, 93, 106, 119]
    );
    let thirty_seconds = super::sampled_frame_indices(720, 24.0, cfg);
    assert_eq!(thirty_seconds.len(), 60);
    assert_eq!(&thirty_seconds[..6], &[0, 12, 24, 37, 49, 61]);
    let ten_minutes = super::sampled_frame_indices(14400, 24.0, cfg);
    assert_eq!(ten_minutes.len(), 768);
    assert_eq!(&ten_minutes[..8], &[0, 19, 38, 56, 75, 94, 113, 131]);
    assert_eq!(ten_minutes[767], 14399);
    // A clip shorter than the minimum gives each of its frames once.
    assert_eq!(super::sampled_frame_indices(1, 30.0, cfg), vec![0]);
    assert_eq!(super::sampled_frame_indices(3, 10.0, cfg), vec![0, 1, 2]);
    // An unknown rate takes the maximum, bounded by the clip.
    assert_eq!(
        super::sampled_frame_indices(5, 0.0, cfg),
        vec![0, 1, 2, 3, 4]
    );
}

#[test]
fn ffmpeg_decodes_carry_the_thread_budget_ahead_of_the_input() {
    let args = |autorotate: bool| {
        super::ffmpeg_decode_command(3, autorotate)
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        args(true),
        [
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "3",
            "-i"
        ]
    );
    assert_eq!(args(false).last().map(String::as_str), Some("-i"));
    assert!(args(false).contains(&"-noautorotate".to_string()));
}

#[test]
fn decoder_threads_share_cpu_budget_across_active_decodes() {
    assert_eq!(super::adaptive_decoder_threads(224, 1), 16);
    assert_eq!(super::adaptive_decoder_threads(2, 1), 4);
    assert_eq!(super::adaptive_decoder_threads(4, 2), 4);
    assert_eq!(super::adaptive_decoder_threads(8, 4), 4);
    assert_eq!(super::adaptive_decoder_threads(8, 8), 1);
    assert_eq!(super::adaptive_decoder_threads(8, 9), 1);
    assert_eq!(super::adaptive_decoder_threads(16, 8), 4);
    assert_eq!(super::adaptive_decoder_threads(16, 16), 1);
    assert_eq!(super::adaptive_decoder_threads(224, 8), 8);
    assert_eq!(super::adaptive_decoder_threads(224, 32), 6);
    assert_eq!(super::adaptive_decoder_threads(8, 32), 1);
    assert_eq!(super::adaptive_decoder_threads(1, 0), 2);
}

/// Defense in depth for the video path: the tracker validates M3's range,
/// but the decoder must reject a zero or off-grid cap itself, before it
/// spends a decode on the clip, exactly as the image decoder does.
#[tokio::test]
async fn decode_video_rejects_an_invalid_long_side_cap() {
    let connector = MediaConnector::new(
        crate::test_support::http_client_builder(),
        MediaConnectorConfig::default(),
    )
    .expect("default connector");
    for value in [0, 505] {
        let cfg = VideoFetchConfig {
            max_long_side_pixel: Some(value),
            ..VideoFetchConfig::default()
        };
        let err = connector
            .fetch_video(MediaSource::InlineBytes(vec![0u8; 16]), cfg)
            .await
            .expect_err("an invalid cap must be rejected");
        assert!(
            matches!(
                err,
                MediaConnectorError::InvalidMaxLongSidePixel { value: got, factor: 28 }
                    if got == value
            ),
            "cap {value}: unexpected error {err:?}"
        );
    }
}
