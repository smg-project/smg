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
    let ffmpeg = executable_starts("ffmpeg");
    let ffprobe = executable_starts("ffprobe");
    let not_found =
        |r: &std::io::Result<()>| matches!(r, Err(e) if e.kind() == std::io::ErrorKind::NotFound);
    match missing_video_decoder() {
        None => assert!(
            ffmpeg.is_ok() && ffprobe.is_ok(),
            "no gap reported while a decoder executable is missing"
        ),
        Some(why) => {
            assert!(
                ffmpeg.is_err() || ffprobe.is_err(),
                "a gap reported while both executables run: {why}"
            );
            assert_eq!(why.contains("ffmpeg "), ffmpeg.is_err(), "{why}");
            assert_eq!(why.contains("ffprobe "), ffprobe.is_err(), "{why}");
            // The install hint is for a binary that is not there; one that is
            // there but cannot run is told that instead.
            assert_eq!(
                why.ends_with("install ffmpeg to decode video_url inputs"),
                not_found(&ffmpeg) || not_found(&ffprobe),
                "{why}"
            );
        }
    }
    assert!(not_found(&executable_starts(
        "an-executable-this-host-does-not-have"
    )));
}
