use std::io::{Error, ErrorKind};

use super::super::*;

/// The probe asks whether `ffmpeg` starts and nothing else: `ffprobe` is
/// optional on the decode path (its probes fall back to `ffmpeg`), so a
/// host with `ffmpeg` alone decodes video. A missing binary gets the install
/// hint; one that is there but cannot run, or exits non-zero when probed, is
/// told that instead.
#[test]
fn the_video_decoder_gap_is_about_ffmpeg_alone() {
    if cfg!(feature = "opencv-video") || std::env::var_os("SMG_VIDEO_DECODE_BACKEND").is_some() {
        return;
    }
    let only_ffmpeg = |program: &str| {
        assert_eq!(program, "ffmpeg", "the probe asked about {program}");
        Ok(())
    };
    assert_eq!(probe_video_decoder_gap(only_ffmpeg), None);
    assert_eq!(
        probe_video_decoder_gap(|_| Err(Error::from(ErrorKind::NotFound))).as_deref(),
        Some("ffmpeg executable not found; install ffmpeg to decode video_url inputs")
    );
    assert_eq!(
        probe_video_decoder_gap(|_| Err(Error::from(ErrorKind::PermissionDenied))).as_deref(),
        Some("ffmpeg could not be started: permission denied")
    );
    assert_eq!(
        probe_video_decoder_gap(|_| Err(Error::other("exited with exit status: 127"))).as_deref(),
        Some("ffmpeg could not be started: exited with exit status: 127")
    );
}

/// On this host the live probe agrees with starting `ffmpeg -version`
/// itself, and a program that is not there is `NotFound`.
#[test]
fn the_live_probe_reads_this_hosts_ffmpeg() {
    if cfg!(feature = "opencv-video") || std::env::var_os("SMG_VIDEO_DECODE_BACKEND").is_some() {
        return;
    }
    let ffmpeg = executable_starts("ffmpeg");
    match missing_video_decoder() {
        None => assert!(
            ffmpeg.is_ok(),
            "no gap reported while ffmpeg does not start"
        ),
        Some(why) => {
            let error = ffmpeg.expect_err("a gap reported while ffmpeg starts");
            assert_eq!(
                why.ends_with("install ffmpeg to decode video_url inputs"),
                error.kind() == ErrorKind::NotFound,
                "{why}"
            );
        }
    }
    assert_eq!(
        executable_starts("an-executable-this-host-does-not-have")
            .expect_err("a program that is not there")
            .kind(),
        ErrorKind::NotFound
    );
}
