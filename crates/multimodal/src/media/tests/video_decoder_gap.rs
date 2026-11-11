use super::super::*;

/// What keeps this process from decoding video is read once from PATH:
/// without the OpenCV backend, `ffmpeg` and `ffprobe` must both start, and
/// the gap names the program that does not. A backend override is not set
/// in the crate's tests, so the probe reads the executables alone.
#[test]
fn the_video_decoder_gap_names_the_executable_that_is_missing() {
    if cfg!(feature = "opencv-video") || std::env::var_os("SMG_VIDEO_DECODE_BACKEND").is_some() {
        return;
    }
    let ffmpeg = executable_runs("ffmpeg");
    let ffprobe = executable_runs("ffprobe");
    match missing_video_decoder() {
        None => assert!(
            ffmpeg && ffprobe,
            "no gap reported while a decoder executable is missing"
        ),
        Some(why) => {
            assert!(
                !(ffmpeg && ffprobe),
                "a gap reported while both executables run: {why}"
            );
            assert_eq!(why.contains("ffmpeg executable"), !ffmpeg, "{why}");
            assert_eq!(why.contains("ffprobe executable"), !ffprobe, "{why}");
            assert!(
                why.ends_with("install ffmpeg to decode video_url inputs"),
                "{why}"
            );
        }
    }
    assert!(!executable_runs("an-executable-this-host-does-not-have"));
}
