//! The video stream metadata the probe tool reports for an MP4 / MOV file,
//! read from the container's own tables.
//!
//! The media connector sizes a clip (its frame count, frame rate and
//! duration) before it decodes it, to pin the frames the engine's loader
//! would pick. That took a probe process per clip; for the common file shape
//! the same numbers sit in the `moov` box and are read here without one. The
//! shape is checked, not assumed: the first video track (the probe's `v:0`;
//! later video tracks are ignored, as the probe ignores them), a single
//! constant frame duration, an edit list that is the whole track at rate 1.0
//! exactly or none, a stated size and a positive duration. Any other file
//! returns `None` and keeps the probe process, so the values this module
//! yields are the probe's for every file it accepts (the agreement test runs
//! the tool when it is installed). One caveat: the size is the sample entry's,
//! what the muxer wrote, which the probe replaces with the decoder's own size
//! after it has looked at the stream; the two agree on the files common
//! muxers write, and the connector checks the decoded frames' own size
//! against this one and re-reads a disagreeing file from the probe before any
//! raw framing:
//!
//! - `width` / `height`: the first visual sample entry's;
//! - `duration_ts` / `time_base`: the media header's duration and time scale
//!   (the stream duration the tool prints is their product, with six decimals);
//! - `nb_frames`: the sample count;
//! - `avg_frame_rate`: the time scale times the sample count over the track
//!   duration, in lowest terms (the demuxer's own derivation);
//! - the movie header's duration and time scale for the container's duration.
//!
//! The edit list may start at the track's first sample or at the codec's
//! reorder delay, where ffmpeg's muxer starts it by default for a video with
//! B-frames; the demuxer reports the same numbers either way (see
//! `edit_list_is_whole_track`).

use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
};

/// The largest `moov` box read into memory; a sane container's is kilobytes
/// per minute of video.
const MAX_MOOV_BYTES: u64 = 64 * 1024 * 1024;

/// What the container says about its first video track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContainerVideoInfo {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The track's media duration, in `timescale` units per second.
    pub(crate) duration_ts: u64,
    pub(crate) timescale: u32,
    /// The average frame rate as a reduced fraction.
    pub(crate) frame_rate: (u64, u64),
    pub(crate) frame_count: u64,
    /// The movie's duration and time scale (the container's duration).
    pub(crate) movie_duration_ts: u64,
    pub(crate) movie_timescale: u32,
}

impl ContainerVideoInfo {
    /// Seconds as the probe tool prints a time: the ticks times the time
    /// base, written with six decimals and read back.
    fn seconds(ticks: u64, timescale: u32) -> Option<f64> {
        if timescale == 0 {
            return None;
        }
        let value = ticks as f64 * (1.0 / f64::from(timescale));
        let printed = format!("{value:.6}");
        printed
            .parse::<f64>()
            .ok()
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
    }

    /// The stream duration in seconds, as the tool reports it.
    pub(crate) fn stream_duration_seconds(&self) -> Option<f64> {
        Self::seconds(self.duration_ts, self.timescale)
    }

    /// The container duration in seconds, as the tool reports it.
    pub(crate) fn movie_duration_seconds(&self) -> Option<f64> {
        Self::seconds(self.movie_duration_ts, self.movie_timescale)
    }

    /// The frame rate as the tool's `num/den` string reads back.
    pub(crate) fn frames_per_second(&self) -> Option<f64> {
        let (num, den) = self.frame_rate;
        let ratio = num as f64 / den as f64;
        (ratio.is_finite() && ratio > 0.0).then_some(ratio)
    }
}

/// Reads the file's container tables; `Ok(None)` when the file is not an
/// MP4 / MOV of the shape described above.
pub(crate) fn read(path: &Path) -> io::Result<Option<ContainerVideoInfo>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let mut offset = 0u64;
    let mut first = true;
    while offset + 8 <= len {
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header[..8])?;
        let mut size = u64::from(u32::from_be_bytes([
            header[0], header[1], header[2], header[3],
        ]));
        let kind = [header[4], header[5], header[6], header[7]];
        let mut header_len = 8u64;
        if size == 1 {
            file.read_exact(&mut header[8..16])?;
            size = u64::from_be_bytes([
                header[8], header[9], header[10], header[11], header[12], header[13], header[14],
                header[15],
            ]);
            header_len = 16;
        } else if size == 0 {
            size = len - offset;
        }
        // `checked_add`: a crafted 64-bit size that wraps the sum would otherwise
        // pass the bound and send the walk back to the start, forever.
        if size < header_len || offset.checked_add(size).is_none_or(|end| end > len) {
            return Ok(None);
        }
        if first {
            if &kind != b"ftyp" {
                return Ok(None);
            }
            first = false;
        }
        if &kind == b"moov" {
            let payload = size - header_len;
            if payload > MAX_MOOV_BYTES {
                return Ok(None);
            }
            let mut moov = vec![0u8; payload as usize];
            file.read_exact(&mut moov)?;
            return Ok(parse_moov(&moov));
        }
        offset += size;
    }
    Ok(None)
}

/// The boxes of `data`: `(kind, payload)` in order; `None` on a malformed size.
fn boxes(data: &[u8]) -> Option<Vec<([u8; 4], &[u8])>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let mut size = u32::from_be_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        let kind: [u8; 4] = data[offset + 4..offset + 8].try_into().ok()?;
        let mut header = 8usize;
        if size == 1 {
            let large = u64::from_be_bytes(data.get(offset + 8..offset + 16)?.try_into().ok()?);
            size = usize::try_from(large).ok()?;
            header = 16;
        } else if size == 0 {
            size = data.len() - offset;
        }
        if size < header || offset.checked_add(size).is_none_or(|end| end > data.len()) {
            return None;
        }
        out.push((kind, &data[offset + header..offset + size]));
        offset += size;
    }
    Some(out)
}

fn child<'a>(list: &[([u8; 4], &'a [u8])], kind: [u8; 4]) -> Option<&'a [u8]> {
    list.iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, payload)| *payload)
}

/// The `kind` box of `list` if it holds at most one: `Some(None)` when it
/// holds none, `None` when it holds several.
fn sole_child<'a>(list: &[([u8; 4], &'a [u8])], kind: [u8; 4]) -> Option<Option<&'a [u8]>> {
    let mut found = list
        .iter()
        .filter(|(k, _)| *k == kind)
        .map(|(_, payload)| *payload);
    let first = found.next();
    found.next().is_none().then_some(first)
}

fn be_u32(data: &[u8], at: usize) -> Option<u32> {
    data.get(at..at + 4)
        .map(|bytes| u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn be_u64(data: &[u8], at: usize) -> Option<u64> {
    data.get(at..at + 8).map(|bytes| {
        u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    })
}

fn be_u16(data: &[u8], at: usize) -> Option<u16> {
    data.get(at..at + 2)
        .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]))
}

/// `(timescale, duration)` of a movie or media header (versions 0 and 1).
fn header_timescale_duration(payload: &[u8]) -> Option<(u32, u64)> {
    match payload.first()? {
        0 => Some((be_u32(payload, 12)?, u64::from(be_u32(payload, 16)?))),
        1 => Some((be_u32(payload, 20)?, be_u64(payload, 24)?)),
        _ => None,
    }
}

/// Whether the demuxer reads `time_scale` as written: it holds a time scale
/// in a signed 32-bit integer and puts 1 in place of one that is not
/// positive (zero, or past `i32::MAX`).
fn time_scale_kept(time_scale: u32) -> bool {
    i32::try_from(time_scale).is_ok_and(|time_scale| time_scale > 0)
}

fn parse_moov(moov: &[u8]) -> Option<ContainerVideoInfo> {
    let top = boxes(moov)?;
    let (movie_timescale, movie_duration_ts) = header_timescale_duration(child(&top, *b"mvhd")?)?;
    // The demuxer puts a tick a second in place of a time scale it cannot
    // hold, so its numbers are not the header's; and against a zero movie
    // time scale a zero-length edit would pass for the track.
    if !time_scale_kept(movie_timescale) {
        return None;
    }
    for (kind, trak) in &top {
        if kind != b"trak" {
            continue;
        }
        let trak = boxes(trak)?;
        let mdia = boxes(child(&trak, *b"mdia")?)?;
        let hdlr = child(&mdia, *b"hdlr")?;
        if hdlr.get(8..12)? != b"vide" {
            continue;
        }
        // The first video track decides; a shape this module does not
        // accept keeps the probe process.
        let (timescale, duration_ts) = header_timescale_duration(child(&mdia, *b"mdhd")?)?;
        if !time_scale_kept(timescale) || duration_ts == 0 {
            return None;
        }
        let stbl = boxes(child(&boxes(child(&mdia, *b"minf")?)?, *b"stbl")?)?;
        let (width, height) = sample_entry_size(child(&stbl, *b"stsd")?)?;
        let stts = child(&stbl, *b"stts")?;
        if be_u32(stts, 4)? != 1 {
            return None;
        }
        let (count, delta) = (u64::from(be_u32(stts, 8)?), u64::from(be_u32(stts, 12)?));
        if count == 0 || delta == 0 || count.checked_mul(delta)? != duration_ts {
            return None;
        }
        // After the sample table: a codec delay is counted in its frames. At
        // most one edit list in at most one `edts`: of several, the demuxer
        // reads the last that has entries.
        if let Some(edts) = sole_child(&trak, *b"edts")? {
            if !edit_list_is_whole_track(
                sole_child(&boxes(edts)?, *b"elst")??,
                duration_ts,
                timescale,
                movie_timescale,
                delta,
            ) {
                return None;
            }
        }
        let sample_count = match (child(&stbl, *b"stsz"), child(&stbl, *b"stz2")) {
            (Some(stsz), _) => u64::from(be_u32(stsz, 8)?),
            (None, Some(stz2)) => u64::from(be_u32(stz2, 8)?),
            (None, None) => return None,
        };
        if sample_count != count {
            return None;
        }
        // The demuxer's average frame rate: time scale x frames / duration, reduced.
        let num = u64::from(timescale).checked_mul(count)?;
        let g = gcd(num, duration_ts);
        return Some(ContainerVideoInfo {
            width,
            height,
            duration_ts,
            timescale,
            frame_rate: (num / g, duration_ts / g),
            frame_count: count,
            movie_duration_ts,
            movie_timescale,
        });
    }
    None
}

/// The deepest codec reorder delay accepted as the start of an edit list, in
/// frames. A muxer starts the list at the delay so that the first frame is
/// presented at zero, and the decoded picture buffer bounds the delay: 16
/// frames in H.264 (`max_num_reorder_frames` <= `max_dec_frame_buffering`
/// <= 16), 15 in HEVC (`sps_max_num_reorder_pics` <=
/// `sps_max_dec_pic_buffering_minus1` <= 15).
const MAX_REORDER_FRAMES: u64 = 16;

/// Whether an edit list that starts `media_time` media ticks into the track
/// starts where a muxer starts one: at the first sample, or at a codec
/// reorder delay of whole frames (`delta` ticks each) up to
/// [`MAX_REORDER_FRAMES`]. A negative start is neither: `-1` is an empty
/// edit, which presents nothing for its duration, and the format defines no
/// other.
fn is_codec_reorder_delay(media_time: i64, delta: u64) -> bool {
    let Ok(media_time) = u64::try_from(media_time) else {
        return false;
    };
    delta != 0 && media_time % delta == 0 && media_time / delta <= MAX_REORDER_FRAMES
}

/// True for an edit list of one entry, at normal speed, whose duration is
/// the track's and which starts at the first sample or at a codec reorder
/// delay: the shapes under which the demuxer's stream duration is the media
/// header's and its frame count the sample count.
///
/// The start is in none of those numbers. The demuxer takes the stream
/// duration from the media header, the sample table and the edit's
/// duration, and the frame count and average rate from the sample table, so
/// a whole-track edit reports the same numbers wherever it starts, and an
/// edit shorter than the track is refused by the duration clause, since the
/// demuxer reports the edit's length. The start is held to what a muxer
/// writes all the same, whole frames up to [`MAX_REORDER_FRAMES`], so that a
/// shape nobody has compared with the probe keeps the probe. Nor is it
/// checked against the track's composition offsets: a start past the codec's
/// own delay drops frames from the decoded output, and the probe still
/// counts them. `delta` is the track's one frame duration, which the caller
/// has read and checked.
fn edit_list_is_whole_track(
    elst: &[u8],
    duration_ts: u64,
    timescale: u32,
    movie_timescale: u32,
    delta: u64,
) -> bool {
    let Some(entries) = be_u32(elst, 4) else {
        return false;
    };
    if entries != 1 {
        return false;
    }
    // The rate is 16.16 fixed point: its integer part and its fraction.
    let (segment_duration, media_time, rate, fraction) = match elst.first() {
        Some(0) => match (
            be_u32(elst, 8),
            be_u32(elst, 12),
            be_u16(elst, 16),
            be_u16(elst, 18),
        ) {
            (Some(d), Some(t), Some(r), Some(f)) => (u64::from(d), i64::from(t as i32), r, f),
            _ => return false,
        },
        Some(1) => match (
            be_u64(elst, 8),
            be_u64(elst, 16),
            be_u16(elst, 24),
            be_u16(elst, 26),
        ) {
            (Some(d), Some(t), Some(r), Some(f)) => (d, t as i64, r, f),
            _ => return false,
        },
        _ => return false,
    };
    // A start a muxer writes (in the media's time scale), rate 1.0 exactly,
    // and the edit's duration (in the movie's time scale) equal to the
    // track's (in the media's). Not every cut reaches the duration clause: a
    // B-frame clip stream-copied with `-t` to a point inside a group of
    // B-frames keeps an edit as long as its media header, which then runs
    // past the sample table, and is refused before, by the header check.
    is_codec_reorder_delay(media_time, delta)
        && rate == 1
        && fraction == 0
        && u128::from(segment_duration) * u128::from(timescale)
            == u128::from(duration_ts) * u128::from(movie_timescale)
}

/// `(width, height)` of the first visual sample entry of an `stsd` box.
fn sample_entry_size(stsd: &[u8]) -> Option<(u32, u32)> {
    if be_u32(stsd, 4)? == 0 {
        return None;
    }
    // After the full-box header and the entry count: the first sample entry,
    // a box whose visual payload carries the size at offsets 24 and 26.
    let entry = &stsd[8..];
    let size = be_u32(entry, 0)? as usize;
    if size < 8 + 28 || size > entry.len() {
        return None;
    }
    let payload = &entry[8..size];
    let width = u32::from(be_u16(payload, 24)?);
    let height = u32::from(be_u16(payload, 26)?);
    (width > 0 && height > 0).then_some((width, height))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(&kind);
        out.extend_from_slice(payload);
        out
    }

    fn full(version: u8, body: &[u8]) -> Vec<u8> {
        let mut out = vec![version, 0, 0, 0];
        out.extend_from_slice(body);
        out
    }

    /// A movie of `frames` frames of `delta` ticks at `timescale`, one video
    /// track, `width` x `height`, an optional edit list.
    fn movie(
        timescale: u32,
        delta: u32,
        frames: u32,
        width: u16,
        height: u16,
        edit: Option<Vec<u8>>,
        stts_entries: Vec<(u32, u32)>,
    ) -> Vec<u8> {
        let edts = edit
            .map(|elst| boxed(*b"edts", &boxed(*b"elst", &elst)))
            .unwrap_or_default();
        movie_with_edts(timescale, delta, frames, width, height, &edts, stts_entries)
    }

    /// `movie`, with `edts` (whole boxes, as many as it holds) where the
    /// track keeps its edit list.
    fn movie_with_edts(
        timescale: u32,
        delta: u32,
        frames: u32,
        width: u16,
        height: u16,
        edts: &[u8],
        stts_entries: Vec<(u32, u32)>,
    ) -> Vec<u8> {
        let duration = u64::from(delta) * u64::from(frames);
        let movie_timescale = 1000u32;
        let movie_duration = (duration * u64::from(movie_timescale) / u64::from(timescale)) as u32;
        let mut mvhd = Vec::new();
        mvhd.extend_from_slice(&[0u8; 8]);
        mvhd.extend_from_slice(&movie_timescale.to_be_bytes());
        mvhd.extend_from_slice(&movie_duration.to_be_bytes());
        mvhd.extend_from_slice(&[0u8; 80]);
        let mut mdhd = Vec::new();
        mdhd.extend_from_slice(&[0u8; 8]);
        mdhd.extend_from_slice(&timescale.to_be_bytes());
        mdhd.extend_from_slice(&(duration as u32).to_be_bytes());
        mdhd.extend_from_slice(&[0u8; 4]);
        let mut hdlr = vec![0u8; 4];
        hdlr.extend_from_slice(b"vide");
        hdlr.extend_from_slice(&[0u8; 13]);
        let mut avc1 = vec![0u8; 24];
        avc1.extend_from_slice(&width.to_be_bytes());
        avc1.extend_from_slice(&height.to_be_bytes());
        avc1.extend_from_slice(&[0u8; 50]);
        let mut stsd = 1u32.to_be_bytes().to_vec();
        stsd.extend_from_slice(&boxed(*b"avc1", &avc1));
        let mut stts = (stts_entries.len() as u32).to_be_bytes().to_vec();
        for (count, d) in &stts_entries {
            stts.extend_from_slice(&count.to_be_bytes());
            stts.extend_from_slice(&d.to_be_bytes());
        }
        let mut stsz = 0u32.to_be_bytes().to_vec();
        stsz.extend_from_slice(&frames.to_be_bytes());
        let stbl = [
            boxed(*b"stsd", &full(0, &stsd)),
            boxed(*b"stts", &full(0, &stts)),
            boxed(*b"stsz", &full(0, &stsz)),
        ]
        .concat();
        let minf = boxed(*b"stbl", &stbl);
        let mdia = [
            boxed(*b"mdhd", &full(0, &mdhd)),
            boxed(*b"hdlr", &full(0, &hdlr)),
            boxed(*b"minf", &minf),
        ]
        .concat();
        let mut trak = edts.to_vec();
        trak.extend_from_slice(&boxed(*b"mdia", &mdia));
        let moov = [boxed(*b"mvhd", &full(0, &mvhd)), boxed(*b"trak", &trak)].concat();
        let mut file = boxed(*b"ftyp", b"isom\0\0\0\0isom");
        file.extend_from_slice(&boxed(*b"mdat", &[7u8; 40]));
        file.extend_from_slice(&boxed(*b"moov", &moov));
        file
    }

    fn whole_track_edit(movie_duration_ms: u32) -> Vec<u8> {
        let mut elst = full(0, &1u32.to_be_bytes());
        elst.extend_from_slice(&movie_duration_ms.to_be_bytes());
        elst.extend_from_slice(&0u32.to_be_bytes());
        elst.extend_from_slice(&1u16.to_be_bytes());
        elst.extend_from_slice(&0u16.to_be_bytes());
        elst
    }

    /// `whole_track_edit`, starting `media_time` media ticks into the track
    /// instead of at its first sample (`-1`: an empty edit).
    fn edit_starting_at(movie_duration_ms: u32, media_time: i32) -> Vec<u8> {
        let mut elst = whole_track_edit(movie_duration_ms);
        elst[12..16].copy_from_slice(&media_time.to_be_bytes());
        elst
    }

    fn written(bytes: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(bytes).expect("write");
        file
    }

    /// A 64-bit box size chosen so that `offset + size` wraps to zero: the
    /// walk rejects the file instead of starting over at the top forever.
    #[test]
    fn a_box_size_that_wraps_the_offset_is_rejected_not_walked() {
        let mut bytes = boxed(*b"ftyp", &[0u8; 8]);
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(b"free");
        bytes.extend_from_slice(&(u64::MAX - 15).to_be_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let file = written(&bytes);
        assert!(read(file.path()).expect("readable").is_none());

        // The same wrap inside a payload: an empty box first, so the 64-bit
        // size starts at offset 8 and `8 + (2^64 - 8)` wraps to zero.
        let mut payload = boxed(*b"free", &[]);
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(b"free");
        payload.extend_from_slice(&(u64::MAX - 7).to_be_bytes());
        assert!(boxes(&payload).is_none());
    }

    /// An edit at rate 1.5 (`0x0001_8000`) is not the whole track at normal
    /// speed, even though its integer part reads 1.
    #[test]
    fn an_edit_at_a_fractional_rate_keeps_the_probe() {
        let mut elst = whole_track_edit(1000);
        let end = elst.len();
        elst[end - 2..].copy_from_slice(&0x8000u16.to_be_bytes());
        let file = written(&movie(
            12288,
            512,
            24,
            320,
            240,
            Some(elst),
            vec![(24, 512)],
        ));
        assert!(read(file.path()).expect("readable").is_none());
    }

    /// A 24 fps track of 24 frames with the whole-track edit list: the
    /// probe's numbers (1.000000 s, 24/1, 24 frames, 320x240).
    #[test]
    fn a_constant_rate_track_is_read_like_the_probe() {
        let file = written(&movie(
            12288,
            512,
            24,
            320,
            240,
            Some(whole_track_edit(1000)),
            vec![(24, 512)],
        ));
        let info = read(file.path()).expect("read").expect("accepted");
        assert_eq!((info.width, info.height), (320, 240));
        assert_eq!((info.duration_ts, info.timescale), (12288, 12288));
        assert_eq!(info.frame_rate, (24, 1));
        assert_eq!(info.frame_count, 24);
        assert_eq!(info.stream_duration_seconds(), Some(1.0));
        assert_eq!(info.movie_duration_seconds(), Some(1.0));
        assert_eq!(info.frames_per_second(), Some(24.0));
    }

    /// NTSC rates reduce the way the demuxer reduces them and print the way
    /// the tool prints them.
    #[test]
    fn fractional_rates_reduce_and_round_like_the_tool() {
        let file = written(&movie(30000, 1001, 300, 1280, 720, None, vec![(300, 1001)]));
        let info = read(file.path()).expect("read").expect("accepted");
        assert_eq!(info.frame_rate, (30000, 1001));
        assert_eq!(info.stream_duration_seconds(), Some(10.01));
        assert_eq!(info.frames_per_second(), Some(30000.0 / 1001.0));
    }

    /// The edit list a muxer writes when the codec reorders frames: the whole
    /// track at normal speed, started at the reorder delay, a whole number of
    /// frames in (16 is the deepest accepted). The demuxer's numbers do not
    /// depend on the start, so the track reads exactly as it does when its
    /// edit list starts at the first sample. (A real B-frame track also
    /// carries composition offsets, `ctts`; the reader does not read them,
    /// and the agreement test in the media module runs real B-frame clips.)
    #[test]
    fn a_codec_delay_edit_is_read_like_the_probe() {
        let undelayed = written(&movie(
            12288,
            512,
            24,
            320,
            240,
            Some(whole_track_edit(1000)),
            vec![(24, 512)],
        ));
        let undelayed = read(undelayed.path()).expect("readable");
        assert!(undelayed.is_some());
        for frames in [1, 2, 4, 16] {
            let file = written(&movie(
                12288,
                512,
                24,
                320,
                240,
                Some(edit_starting_at(1000, 512 * frames)),
                vec![(24, 512)],
            ));
            assert_eq!(
                read(file.path()).expect("readable"),
                undelayed,
                "a {frames}-frame codec delay"
            );
        }
    }

    /// Starts a muxer does not write for a codec delay keep the probe
    /// process: deeper than 16 frames, part of a frame in, or before the
    /// track (`-1` is an empty edit, which presents nothing for its
    /// duration; the format defines no other negative start).
    #[test]
    fn an_edit_that_is_no_codec_delay_keeps_the_probe() {
        for media_time in [512 * 17, 700, -1, -512] {
            let file = written(&movie(
                12288,
                512,
                24,
                320,
                240,
                Some(edit_starting_at(1000, media_time)),
                vec![(24, 512)],
            ));
            assert_eq!(
                read(file.path()).expect("readable"),
                None,
                "an edit list starting at {media_time}"
            );
        }
    }

    /// An edit list that starts at a codec delay but is not the one whole
    /// track at normal speed keeps the probe process. A shortened edit
    /// changes what the demuxer reports: the edit's length, which the tables
    /// do not show. The rate and the entry count are shape policy (the
    /// demuxer ignores the rate, and reports these two-entry lists as it
    /// reports the track): the module vouches only for the shape it has
    /// compared with the probe.
    #[test]
    fn an_edit_that_is_not_the_whole_track_keeps_the_probe() {
        // 0.9 s of the 1 s track: a trim the tables do not show.
        let shortened = edit_starting_at(900, 1024);
        // Rate 2.0 (`0x0002_0000`), whose fraction is zero.
        let mut doubled = edit_starting_at(1000, 1024);
        doubled[16..18].copy_from_slice(&2u16.to_be_bytes());
        // Two entries: 83 ms of nothing (an empty edit) and then the track,
        // as `-avoid_negative_ts make_zero` writes it, which its first start
        // declines as well; and the whole track followed by half of it
        // again, which only the entry count declines.
        let two_entries = |entries: [(u32, i32); 2]| {
            let mut elst = full(0, &2u32.to_be_bytes());
            for (duration, start) in entries {
                elst.extend_from_slice(&duration.to_be_bytes());
                elst.extend_from_slice(&start.to_be_bytes());
                elst.extend_from_slice(&[0, 1, 0, 0]);
            }
            elst
        };
        let cases = [
            ("shortened", shortened),
            ("rate 2", doubled),
            ("an empty edit first", two_entries([(83, -1), (1000, 1024)])),
            ("a second edit", two_entries([(1000, 1024), (500, 1024)])),
        ];
        for (case, elst) in cases {
            let file = written(&movie(
                12288,
                512,
                24,
                320,
                240,
                Some(elst),
                vec![(24, 512)],
            ));
            assert_eq!(read(file.path()).expect("readable"), None, "{case}");
        }
    }

    /// The repository's H.264 fixture starts its edit list at a codec delay,
    /// but its media header runs one frame past its sample table (18432
    /// ticks against 8 frames of 2048), a shape ffmpeg's muxer writes for
    /// some B-frame clips. The demuxer then reports the shorter of the two,
    /// which this module does not compute, so the file keeps the probe. The
    /// fixture's edit is as long as its samples, so the duration clause
    /// declines it too; a copy whose edit is as long as the header, as a
    /// B-frame clip cut with `-t` inside a group of B-frames has it, is
    /// declined by the header check alone.
    #[test]
    fn a_media_header_past_the_samples_keeps_the_probe() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/videos/testsrc_128x96_8fps_1s.mp4");
        assert_eq!(read(&fixture).expect("readable"), None);
        let mut bytes = std::fs::read(&fixture).expect("fixture bytes");
        // The edit's duration, in the movie's milliseconds: after the `elst`
        // box type, its version, flags and entry count.
        let elst = bytes
            .windows(4)
            .position(|kind| kind == b"elst")
            .expect("elst")
            + 4;
        assert_eq!(&bytes[elst + 8..elst + 12], &1000u32.to_be_bytes());
        bytes[elst + 8..elst + 12].copy_from_slice(&1125u32.to_be_bytes());
        assert_eq!(read(written(&bytes).path()).expect("readable"), None);
    }

    /// Two edit lists in one track, in one `edts` or in two: the demuxer
    /// reads the last that has entries, so the first is no account of the
    /// track. Here the first is the whole track and the last only half of
    /// it, the half the demuxer would report; the file keeps the probe
    /// process.
    #[test]
    fn a_track_with_two_edit_lists_keeps_the_probe() {
        let whole = boxed(*b"elst", &edit_starting_at(1000, 512));
        let half = boxed(*b"elst", &edit_starting_at(500, 512));
        let cases = [
            (
                "one edts",
                boxed(*b"edts", &[whole.clone(), half.clone()].concat()),
            ),
            (
                "two edts",
                [boxed(*b"edts", &whole), boxed(*b"edts", &half)].concat(),
            ),
        ];
        for (case, edts) in cases {
            let file = written(&movie_with_edts(
                12288,
                512,
                24,
                320,
                240,
                &edts,
                vec![(24, 512)],
            ));
            assert_eq!(read(file.path()).expect("readable"), None, "{case}");
        }
    }

    /// A time scale the demuxer does not read as written keeps the probe
    /// process: it holds a time scale in a signed 32-bit integer and puts 1
    /// in place of one that is not positive. Against a zero movie time scale
    /// a zero-length edit would pass for the whole track, and a media time
    /// scale past `i32::MAX` would be read as written here and as 1 by the
    /// probe.
    #[test]
    fn a_time_scale_the_demuxer_replaces_keeps_the_probe() {
        let cases = [
            (*b"mvhd", 0, Some(edit_starting_at(0, 512))),
            (*b"mvhd", 0x8000_0000, None),
            (*b"mdhd", 0x8000_0000, None),
        ];
        for (header, time_scale, edit) in cases {
            let mut bytes = movie(12288, 512, 24, 320, 240, edit, vec![(24, 512)]);
            // The time scale follows the header's box header, its version and
            // flags, and two 32-bit times.
            let at = bytes
                .windows(4)
                .position(|kind| kind == header)
                .expect("header")
                + 4;
            bytes[at + 12..at + 16].copy_from_slice(&u32::to_be_bytes(time_scale));
            assert_eq!(
                read(written(&bytes).path()).expect("readable"),
                None,
                "{} time scale {time_scale}",
                String::from_utf8_lossy(&header)
            );
        }
    }

    /// Shapes the module does not vouch for keep the probe process.
    #[test]
    fn other_shapes_are_declined() {
        let vfr = movie(12288, 512, 25, 320, 240, None, vec![(24, 512), (1, 1024)]);
        assert_eq!(read(written(&vfr).path()).expect("read"), None);
        assert_eq!(
            read(written(b"RIFF\0\0\0\0WEBPVP8 ").path()).expect("read"),
            None
        );
        assert_eq!(read(written(&[]).path()).expect("read"), None);
    }
}
