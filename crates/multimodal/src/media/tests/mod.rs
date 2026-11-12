mod decode_limit;
mod exif_orientation;
mod hop_budget;
mod ico_png_entry;
mod max_long_side_pixel;
mod seek_decode;
mod sixteen_bit_colour;
mod sixteen_bit_grey;
mod transparency;
mod video_decoder_gap;
mod video_frame_cap;
mod video_sampling;

use std::{
    collections::HashMap,
    io::Write as _,
    path::{Path, PathBuf},
    process::Command,
};

use crate::test_support::{skip_for, video_tools_available};

/// Whether this ffmpeg build carries `encoder`. `ffmpeg -h encoder=<name>`
/// exits 0 for a name it does not know as well, so the answer is what it
/// prints: the help of a built encoder opens with `Encoder <name> `.
fn have_encoder(encoder: &str) -> bool {
    let built = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-h"])
        .arg(format!("encoder={encoder}"))
        .output()
        .is_ok_and(|out| {
            out.status.success()
                && out
                    .stdout
                    .starts_with(format!("Encoder {encoder} ").as_bytes())
        });
    if !built {
        skip_for(&format!("this ffmpeg has no {encoder} encoder"));
    }
    built
}

/// A `seconds`-long clip of the test pattern at `rate`, written into `dir` as
/// a `container` file with `codec` (the arguments after `-c:v`).
fn write_clip(dir: &Path, codec: &[&str], rate: &str, seconds: u32, container: &str) -> PathBuf {
    let path = dir.join(format!(
        "clip_{}_{}_{seconds}.{container}",
        codec.concat(),
        rate.replace('/', "_")
    ));
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
        .args(["-pix_fmt", "yuv420p", "-c:v"])
        .args(codec)
        .arg(&path)
        .output()
        .expect("ffmpeg runs");
    assert!(
        encoded.status.success(),
        "ffmpeg wrote {path:?}: {}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    path
}

/// The probe tool's report for `path`, as the connector parses it.
fn probe_report(path: &Path) -> ProbedVideoInfo {
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
        .arg(path)
        .output()
        .expect("ffprobe runs");
    assert!(output.status.success());
    parse_ffprobe_video_info(&output.stdout).expect("probe parses")
}

/// Where a clip's edit list starts presenting its track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditStart {
    /// At the first sample: the encoder keeps the frames in order.
    FirstSample,
    /// A whole number of frames in: the encoder writes B-frames, and the
    /// muxer starts the edit list at the codec's reorder delay so that the
    /// first frame is presented at zero.
    CodecDelay,
}

/// Where the first track of the file at `path` starts its edit list, read
/// from its boxes; a start that is neither (before the track, or part of a
/// frame in) fails the test.
fn edit_start(path: &Path) -> EditStart {
    let bytes = std::fs::read(path).expect("clip bytes");
    let (elst, _) = video_sampling::mp4_payload(&bytes, &[*b"moov", *b"trak", *b"edts", *b"elst"]);
    // After the version, flags and entry count: the first entry's duration,
    // then its start, both 32-bit in version 0 and 64-bit in version 1.
    let start = if bytes[elst] == 0 {
        i64::from(i32::from_be_bytes(
            bytes[elst + 12..elst + 16].try_into().unwrap(),
        ))
    } else {
        i64::from_be_bytes(bytes[elst + 16..elst + 24].try_into().unwrap())
    };
    let (stts, _) = video_sampling::mp4_payload(
        &bytes,
        &[*b"moov", *b"trak", *b"mdia", *b"minf", *b"stbl", *b"stts"],
    );
    // After the version, flags, entry count and the first entry's count:
    // its frame duration.
    let frame = i64::from(u32::from_be_bytes(
        bytes[stts + 12..stts + 16].try_into().unwrap(),
    ));
    match start {
        0 => EditStart::FirstSample,
        start if start > 0 && start % frame == 0 => EditStart::CodecDelay,
        start => panic!("{path:?} starts its edit list {start} ticks in, frames of {frame}"),
    }
}

/// Where the tools are installed: clips written by the encoder at several
/// frame rates and lengths, read through the container tables, equal the
/// probe tool's own report field by field (the numbers the frame
/// selection and the clip's sample rate are built from).
///
/// The plain `mpeg4` clips keep their frames in order, so the muxer starts
/// their edit list at the first sample. The B-frame clips (`mpeg4 -bf 2`,
/// and `libx264` and `libx265` at their defaults) start it at the codec's
/// reorder delay, one or two frames in: the shape of a clip from an
/// ordinary encoder. Each clip's start is checked, so an encoder or muxer
/// default that stopped writing the delay fails here rather than leaving
/// those cases to test the first shape again.
#[test]
fn container_tables_agree_with_the_probe_tool() {
    use EditStart::{CodecDelay, FirstSample};
    if !video_tools_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    // (the arguments after `-c:v`, rate, seconds, container, edit list start)
    let cases: [(&[&str], &str, u32, &str, EditStart); 10] = [
        (&["mpeg4"], "24", 1, "mp4", FirstSample),
        (&["mpeg4"], "30000/1001", 2, "mp4", FirstSample),
        (&["mpeg4"], "25", 3, "mov", FirstSample),
        (&["mpeg4"], "60", 1, "mp4", FirstSample),
        (&["mpeg4"], "24000/1001", 2, "mp4", FirstSample),
        (&["mpeg4", "-bf", "2"], "24", 2, "mp4", CodecDelay),
        (&["mpeg4", "-bf", "2"], "30000/1001", 2, "mov", CodecDelay),
        (&["libx264"], "24", 2, "mp4", CodecDelay),
        (&["libx264"], "30000/1001", 2, "mov", CodecDelay),
        (&["libx265"], "25", 3, "mp4", CodecDelay),
    ];
    let mut built = HashMap::new();
    for (codec, rate, seconds, container, start) in cases {
        if !*built
            .entry(codec[0])
            .or_insert_with(|| have_encoder(codec[0]))
        {
            continue;
        }
        let case = format!("{}, {rate} fps, {seconds} s, {container}", codec.join(" "));
        let path = write_clip(dir.path(), codec, rate, seconds, container);
        assert_eq!(edit_start(&path), start, "{case}");
        let container_info = crate::mp4_metadata::read(&path)
            .expect("readable")
            .unwrap_or_else(|| panic!("{case}: not accepted by the container reader"));
        assert_eq!(
            super::probed_from_container(&container_info),
            probe_report(&path),
            "{case}"
        );
    }
}

/// Why the container reader may accept an edit list that starts at the
/// codec delay: the probe reports a clip written with B-frames exactly as
/// it reports the same clip written without them, whose edit list starts at
/// the first sample. The start enters none of the numbers it prints.
#[test]
fn the_codec_delay_changes_nothing_the_probe_reports() {
    if !video_tools_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    // (with B-frames, without), the arguments after `-c:v`
    let pairs: [(&[&str], &[&str]); 2] = [
        (&["mpeg4", "-bf", "2"], &["mpeg4"]),
        (&["libx264"], &["libx264", "-bf", "0"]),
    ];
    for (delayed, control) in pairs {
        if !have_encoder(delayed[0]) {
            continue;
        }
        let delayed_clip = write_clip(dir.path(), delayed, "24", 2, "mp4");
        let control_clip = write_clip(dir.path(), control, "24", 2, "mp4");
        assert_eq!(
            edit_start(&delayed_clip),
            EditStart::CodecDelay,
            "{}",
            delayed.join(" ")
        );
        assert_eq!(
            edit_start(&control_clip),
            EditStart::FirstSample,
            "{}",
            control.join(" ")
        );
        assert_eq!(
            probe_report(&delayed_clip),
            probe_report(&control_clip),
            "{}",
            delayed.join(" ")
        );
    }
}

/// The edit's duration is what keeps a shortened edit list off the
/// container tables. The clip starts its edit list at the codec delay and is
/// read from its tables; then only the edit's duration is cut in place, from
/// the whole 2 s to 1.75 s, with the media header and the sample table left
/// as they were. The probe now reports 1.75 s where the tables still say
/// 2 s, and the reader declines the file, so the probe process runs.
#[test]
fn a_shortened_edit_keeps_the_probe_process() {
    if !video_tools_available() || !have_encoder("mpeg4") {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_clip(dir.path(), &["mpeg4", "-bf", "2"], "24", 2, "mp4");
    assert_eq!(edit_start(&path), EditStart::CodecDelay);
    assert!(
        crate::mp4_metadata::read(&path)
            .expect("readable")
            .is_some(),
        "the whole clip is read from its tables"
    );
    let mut bytes = std::fs::read(&path).expect("clip bytes");
    let (elst, _) = video_sampling::mp4_payload(&bytes, &[*b"moov", *b"trak", *b"edts", *b"elst"]);
    // A version 0 list: after the version, flags and entry count, the first
    // entry's duration in the movie's time scale (milliseconds, as ffmpeg
    // writes it).
    assert_eq!(bytes[elst], 0);
    assert_eq!(&bytes[elst + 8..elst + 12], &2000u32.to_be_bytes());
    bytes[elst + 8..elst + 12].copy_from_slice(&1750u32.to_be_bytes());
    std::fs::write(&path, bytes).expect("clip rewritten");
    assert_eq!(probe_report(&path).duration_seconds, Some(1.75));
    assert_eq!(
        crate::mp4_metadata::read(&path).expect("readable"),
        None,
        "a shortened edit must keep the probe process"
    );
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

/// The frames the Qwen2-VL line's own video loader picks at its 2 fps rate,
/// as that loader computes them (`torch.arange(0, total, total / count)`
/// over `total / fps x 2` frames clamped to [4, 768], the count and the cap
/// aligned down to whole temporal groups of two): the expected picks are
/// that loader's code run on each clip's length and rate. A clip shorter
/// than one group gets no frame, and so does a clip whose rate is unknown,
/// which the loader refuses.
#[test]
fn stepped_group_sampling_picks_the_engine_loaders_frames() {
    let cfg = VideoFetchConfig {
        sampling: FrameSampling::SteppedGroups {
            temporal_patch_size: 2,
            max_frames: 768,
        },
        ..VideoFetchConfig::default()
    };
    let picks = |total: usize, fps: f64| super::sampled_frame_indices(total, fps, cfg);
    assert_eq!(picks(30, 30.0), vec![0, 7, 15, 22], "30 frames at 30 fps");
    assert_eq!(picks(15, 30.0), vec![0, 3, 7, 11], "15 frames at 30 fps");
    assert_eq!(picks(5, 1.0), vec![0, 1, 2, 3], "5 frames at 1 fps");
    assert_eq!(
        picks(150, 30.0),
        vec![0, 15, 30, 45, 60, 75, 90, 105, 120, 135],
        "150 frames at 30 fps"
    );
    let long = picks(451, 30.0);
    assert_eq!(long.len(), 30, "451 frames at 30 fps");
    assert_eq!(&long[..8], &[0, 15, 30, 45, 60, 75, 90, 105]);
    assert_eq!(long[29], 435);
    assert_eq!(picks(24, 24.0), vec![0, 6, 12, 18], "24 frames at 24 fps");
    assert_eq!(
        picks(120, 24.0),
        vec![0, 12, 24, 36, 48, 60, 72, 84, 96, 108],
        "120 frames at 24 fps"
    );
    let long = picks(720, 24.0);
    assert_eq!(long.len(), 60, "720 frames at 24 fps");
    assert_eq!(&long[..8], &[0, 12, 24, 36, 48, 60, 72, 84]);
    assert_eq!(long[59], 708);
    let long = picks(14400, 24.0);
    assert_eq!(long.len(), 768, "14400 frames at 24 fps");
    assert_eq!(&long[..8], &[0, 18, 37, 56, 75, 93, 112, 131]);
    assert_eq!(long[767], 14381);
    assert_eq!(picks(3, 10.0), vec![0, 1], "3 frames at 10 fps");
    assert_eq!(picks(1, 30.0), Vec::<usize>::new(), "1 frame at 30 fps");
    assert_eq!(picks(2, 30.0), vec![0, 1], "2 frames at 30 fps");
    assert_eq!(picks(7, 2.5), vec![0, 1, 3, 5], "7 frames at 2.5 fps");
    assert_eq!(picks(60, 30.0), vec![0, 15, 30, 45], "60 frames at 30 fps");
    let long = picks(300, 30.0);
    assert_eq!(long.len(), 20, "300 frames at 30 fps");
    assert_eq!(&long[..8], &[0, 15, 30, 45, 60, 75, 90, 105]);
    assert_eq!(long[19], 285);
    let long = picks(9000, 30.0);
    assert_eq!(long.len(), 600, "9000 frames at 30 fps");
    assert_eq!(&long[..8], &[0, 15, 30, 45, 60, 75, 90, 105]);
    assert_eq!(long[599], 8985);
    let long = picks(18000, 30.0);
    assert_eq!(long.len(), 768, "18000 frames at 30 fps");
    assert_eq!(&long[..8], &[0, 23, 46, 70, 93, 117, 140, 164]);
    assert_eq!(long[767], 17976);
    let long = picks(36000, 60.0);
    assert_eq!(long.len(), 768, "36000 frames at 60 fps");
    assert_eq!(&long[..8], &[0, 46, 93, 140, 187, 234, 281, 328]);
    assert_eq!(long[767], 35953);
    assert_eq!(
        picks(10, 1.0),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "10 frames at 1 fps"
    );
    assert_eq!(picks(4, 4.0), vec![0, 1, 2, 3], "4 frames at 4 fps");
    assert_eq!(picks(6, 1.0), vec![0, 1, 2, 3, 4, 5], "6 frames at 1 fps");
    assert_eq!(
        picks(11, 1.0),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "11 frames at 1 fps"
    );
    assert_eq!(
        picks(23, 23.976),
        vec![0, 5, 11, 17],
        "23 frames at 23.976 fps"
    );
    let long = picks(1001, 29.97);
    assert_eq!(long.len(), 66, "1001 frames at 29.97 fps");
    assert_eq!(&long[..8], &[0, 15, 30, 45, 60, 75, 91, 106]);
    assert_eq!(long[65], 985);
    // An unknown rate picks nothing: the loader refuses such a clip.
    assert_eq!(picks(150, 0.0), Vec::<usize>::new());
    // The rule follows the loader: the container's claim, cut to the stream.
    assert!(cfg.sampling.follows_the_loader());
}

/// Under the stepped groups a decoded clip's rate is the one the engine's
/// loader reports for its picks, `picks / total x source rate`, which the
/// processor turns into the seconds per temporal group: 4 of 5 frames at 1
/// fps are 0.8 fps (2.5 s a group), where the target-rate estimate says 2
/// fps; 10 of 150 at 30 fps are the target 2 fps. The other rules keep the
/// estimate.
#[test]
fn stepped_group_clips_report_the_loaders_sampled_rate() {
    let stepped = VideoFetchConfig {
        sampling: FrameSampling::SteppedGroups {
            temporal_patch_size: 2,
            max_frames: 768,
        },
        ..VideoFetchConfig::default()
    };
    assert_eq!(super::loader_sampled_fps(4, 5, 1.0), 0.8);
    assert_eq!(super::loader_sampled_fps(10, 150, 30.0), 2.0);
    assert_eq!(super::loader_sampled_fps(768, 14400, 24.0), 1.28);
    assert_eq!(super::loader_sampled_fps(0, 0, 30.0), 30.0);
    let short = super::FrameSelection::new(1.0, 5, 5, vec![0, 1, 2, 3]);
    assert_eq!(
        super::clip_sample_fps(Some(&short), Some(5.0), stepped),
        0.8
    );
    assert_eq!(effective_sample_fps(Some(5.0), stepped), 2.0);
    assert_eq!(super::clip_sample_fps(None, Some(5.0), stepped), 2.0);
    let even = VideoFetchConfig {
        sampling: FrameSampling::EvenRounded,
        ..VideoFetchConfig::default()
    };
    assert_eq!(super::clip_sample_fps(Some(&short), Some(5.0), even), 2.0);
}

/// The interval walk's clips report the rate of its picks the same way: a 1 s
/// clip at 30 fps gives `[0, 29]`, 2 fps (1 s a group of two), where the
/// target-rate estimate under the connector's bounds says 4 fps; an 800 s clip
/// at 1 fps gives 800 picks, 1 fps, where the estimate says 0.96. Without a
/// selection the estimate stands.
#[test]
fn interval_clips_report_the_loaders_sampled_rate() {
    let interval = VideoFetchConfig {
        sampling: FrameSampling::Interval,
        ..VideoFetchConfig::default()
    };
    let one_second = super::FrameSelection::new(30.0, 30, 30, vec![0, 29]);
    assert_eq!(
        super::clip_sample_fps(Some(&one_second), Some(1.0), interval),
        2.0
    );
    assert_eq!(effective_sample_fps(Some(1.0), interval), 4.0);
    let long = super::FrameSelection::new(1.0, 800, 800, (0..800).collect());
    assert_eq!(
        super::clip_sample_fps(Some(&long), Some(800.0), interval),
        1.0
    );
    assert_eq!(effective_sample_fps(Some(800.0), interval), 0.96);
    assert_eq!(super::clip_sample_fps(None, Some(1.0), interval), 4.0);
}

/// A container's frame claim sizes the interval walk's picks. A claim whose
/// picks at source resolution cannot fit the decoded-bytes budget is refused
/// with that budget's own error before any pick is held, and the count stops
/// at the budget, so the claim's size is no cost: two billion frames at 30
/// fps in full HD are refused after the budget's 172 picks, and the same
/// claim in one-pixel frames or from an unsized probe after the picks the
/// budget holds in records alone. A clip within the budget, the other rules
/// (bounded by their caps) and an unknown rate (the bounded `Even` fallback)
/// pass.
#[test]
fn an_interval_claim_past_the_decoded_bytes_budget_is_refused_before_its_picks() {
    let interval = VideoFetchConfig {
        sampling: FrameSampling::Interval,
        ..VideoFetchConfig::default()
    };
    let full_hd = 1920 * 1080 * 3;
    let budget = super::video_max_decoded_bytes();
    for (total_frames, fps) in [(1usize << 31, 30.0), (1_000, 1.0)] {
        let refused =
            super::ensure_interval_picks_fit(total_frames, fps, full_hd, interval, budget);
        assert!(
            matches!(
                refused,
                Err(MediaConnectorError::DecodedTooLarge {
                    media: "video RGB",
                    knob: "SMG_VIDEO_MAX_DECODED_BYTES",
                    ..
                })
            ),
            "{total_frames} frames at {fps} fps: {refused:?}"
        );
    }
    // The smallest frames and an unsized probe are bounded through the picks'
    // records alone: the claim is refused, a clip within the budget passes.
    for frame_size in [3, 0] {
        let refused = super::ensure_interval_picks_fit(1 << 31, 30.0, frame_size, interval, budget);
        assert!(
            matches!(refused, Err(MediaConnectorError::DecodedTooLarge { .. })),
            "frame size {frame_size}: {refused:?}"
        );
        assert!(super::ensure_interval_picks_fit(1_000, 1.0, frame_size, interval, budget).is_ok());
    }
    assert!(super::ensure_interval_picks_fit(150, 30.0, full_hd, interval, budget).is_ok());
    let even = VideoFetchConfig::default();
    assert!(super::ensure_interval_picks_fit(1 << 31, 30.0, full_hd, even, budget).is_ok());
    assert!(super::ensure_interval_picks_fit(1 << 31, f64::NAN, full_hd, interval, budget).is_ok());
    // The boundary, pinned with a small budget and no allocation: at 1 fps
    // every frame is a pick, so the count is the frame count; the records
    // alone price an unsized pick at 128 bytes, a one-pixel pick at 131.
    let small = 1usize << 20;
    for (frame_size, fits, refused) in [(0, 8_192, 8_193), (3, 8_004, 8_005)] {
        assert!(super::ensure_interval_picks_fit(fits, 1.0, frame_size, interval, small).is_ok());
        let error = super::ensure_interval_picks_fit(refused, 1.0, frame_size, interval, small);
        assert!(
            matches!(error, Err(MediaConnectorError::DecodedTooLarge { limit, .. }) if limit == small),
            "frame size {frame_size}: {error:?}"
        );
    }
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
