//! Sampling a long clip by seeking to its picks.
//!
//! The media connector pins the exact source frames a family's rule picks
//! and has the decoder run through the stream from its first frame to the
//! last pick, keeping only those: a clip then costs its whole length however
//! few frames the model takes from it, and half an hour of video is tens of
//! thousands of decoded frames for a few dozen picks.
//!
//! This module plans the decode the other way round. Each pick is decoded
//! from the random-access frame (keyframe) before it, in one decoder run per
//! keyframe that owns picks, so the work is the picks' own neighbourhoods.
//! The plan is built from the stream's packet listing as the probe tool
//! prints it without decoding anything (presentation timestamps and keyframe
//! flags), and it is used only where it clearly beats the run through the
//! stream. The picks are named to the decoder by their exact presentation
//! timestamps, and each run ends right after its last pick: a seek that lands
//! anywhere but where the plan says leaves a pick missing, which costs the run
//! its own segment and no more, and which the caller sees as a short decode
//! and answers with the sequential decode, never with a wrong frame.

/// What one decoder run costs besides the frames it decodes (starting the
/// process, reading the container's tables, seeking), in source frames.
pub(crate) const RUN_OVERHEAD_FRAMES: usize = 64;

/// Plans with more runs than this are not made: the sequential decode is
/// the right tool for a sampling that dense.
pub(crate) const MAX_SEGMENTS: usize = 1024;

/// A time base finer than this makes the microsecond seek position of a
/// frame ambiguous between two ticks; such streams keep the sequential decode.
const MAX_TIME_BASE_DEN: u64 = 500_000;

/// The frames of a video stream in presentation order, read off the probe
/// tool's packet listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FrameTable {
    /// Presentation timestamp in ticks of `time_base` and the keyframe flag,
    /// one entry per frame the decoder outputs, ascending.
    frames: Vec<(i64, bool)>,
    /// The stream time base as `num / den` seconds per tick.
    time_base: (u64, u64),
    /// The container's start time in microseconds, which the decoder tool
    /// measures its seek position from.
    start_time_us: i64,
}

impl FrameTable {
    /// The table from the probe tool's CSV listing of
    /// `packet=pts,flags:stream=time_base:format=start_time` (section names
    /// kept), or `None` when the listing does not describe a stream this
    /// module can plan for: a packet without a timestamp, a corrupt packet,
    /// two frames at one timestamp, no time base or a finer one than the
    /// seek position resolves, or a first frame that is no keyframe. Packets
    /// the demuxer flags for discard (edit list lead-in) are not frames the
    /// decoder outputs and are left out.
    pub(crate) fn parse(listing: &[u8]) -> Option<Self> {
        let listing = std::str::from_utf8(listing).ok()?;
        let mut frames = Vec::new();
        let mut time_base = None;
        let mut start_time_us = None;
        for line in listing.lines() {
            let line = line.trim_end_matches(',');
            let mut fields = line.split(',');
            match fields.next()? {
                "packet" => {
                    let pts = fields.next()?.parse::<i64>().ok()?;
                    let flags = fields.next()?;
                    if flags.contains('C') {
                        return None;
                    }
                    if flags.contains('D') {
                        continue;
                    }
                    frames.push((pts, flags.contains('K')));
                }
                "stream" => {
                    let (num, den) = fields.next()?.split_once('/')?;
                    let parsed = (num.parse::<u64>().ok()?, den.parse::<u64>().ok()?);
                    if parsed.0 == 0 || parsed.1 == 0 {
                        return None;
                    }
                    // A container with programs lists its stream twice.
                    match time_base {
                        None => time_base = Some(parsed),
                        Some(seen) if seen == parsed => {}
                        Some(_) => return None,
                    }
                }
                "format" => {
                    let start = fields.next()?;
                    // No start time: the decoder tool measures from zero.
                    start_time_us = Some(if start == "N/A" {
                        0
                    } else {
                        (start.parse::<f64>().ok()? * 1_000_000.0).round() as i64
                    });
                }
                _ => {}
            }
        }
        let time_base = time_base?;
        if time_base.1 > MAX_TIME_BASE_DEN {
            return None;
        }
        frames.sort_by_key(|(pts, _)| *pts);
        if frames.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return None;
        }
        if !matches!(frames.first(), Some((_, true))) {
            return None;
        }
        Some(Self {
            frames,
            time_base,
            start_time_us: start_time_us.unwrap_or(0),
        })
    }

    /// Frames the decoder outputs.
    pub(crate) fn len(&self) -> usize {
        self.frames.len()
    }

    /// The seek position of the frame at `index`, in seconds from the
    /// container's start with six decimals, as the decoder tool's input seek
    /// option reads it: the frame's own timestamp to the nearest microsecond,
    /// which the tool resolves back to the frame's own tick.
    fn seek_seconds(&self, index: usize) -> String {
        let (num, den) = self.time_base;
        let ticks = i128::from(self.frames[index].0);
        let scaled = ticks * i128::from(num) * 1_000_000;
        let den = i128::from(den);
        // Round half up, for negative timestamps as well.
        let absolute_us = (2 * scaled + den).div_euclid(2 * den);
        let relative_us = absolute_us - i128::from(self.start_time_us);
        let sign = if relative_us < 0 { "-" } else { "" };
        let magnitude = relative_us.abs();
        format!(
            "{sign}{}.{:06}",
            magnitude / 1_000_000,
            magnitude % 1_000_000
        )
    }
}

/// One decoder run of a [`SeekPlan`]: from a keyframe through the picks it owns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SeekSegment {
    /// Where the decoder starts, as its input seek option takes it.
    pub(crate) seek_seconds: String,
    /// The picks' presentation timestamps, ascending.
    pub(crate) pts: Vec<i64>,
    /// Source frames the decoder runs through: the keyframe to the last pick.
    pub(crate) decoded_frames: usize,
}

impl SeekSegment {
    /// The run's frame filter: the stream trimmed right after the last pick,
    /// then the picks named by timestamp. The trim ends the filter graph, and
    /// with it the run, as soon as the decoder is past the last pick, so a
    /// pick the seek missed costs the run its segment and no more; the
    /// `select` alone would have the decoder run to the end of the stream
    /// looking for it.
    pub(crate) fn frame_filter(&self) -> String {
        format!("trim=end_pts={},{}", self.end_pts(), self.select_filter())
    }

    /// The first timestamp past the last pick, in the stream's ticks: where
    /// the run ends.
    pub(crate) fn end_pts(&self) -> i64 {
        self.pts.last().map_or(0, |pts| pts.saturating_add(1))
    }

    /// The frame filter naming the picks by timestamp; `\,` keeps the
    /// filtergraph parser from splitting the expression.
    pub(crate) fn select_filter(&self) -> String {
        let terms: Vec<String> = self
            .pts
            .iter()
            .map(|pts| format!("eq(pts\\,{pts})"))
            .collect();
        format!("select='{}'", terms.join("+"))
    }
}

/// The decoder runs that produce a selection's distinct frames by seeking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SeekPlan {
    pub(crate) segments: Vec<SeekSegment>,
    /// The last pick's index: the frames the sequential decode runs through, less one.
    last_pick: usize,
}

impl SeekPlan {
    /// The plan for `picks` (distinct source indices, ascending) over a
    /// stream of `total_frames` that `table` lists, or `None` when the table
    /// is not that stream (another frame count), a pick is past its end, or
    /// the sampling is too dense for a plan.
    pub(crate) fn build(table: &FrameTable, total_frames: usize, picks: &[usize]) -> Option<Self> {
        if table.len() != total_frames || picks.is_empty() {
            return None;
        }
        let last_pick = *picks.last()?;
        if last_pick >= total_frames {
            return None;
        }
        let mut segments: Vec<SeekSegment> = Vec::new();
        let mut segment_keyframe = None;
        let mut keyframe = 0usize;
        let mut cursor = 0usize;
        for &pick in picks {
            // The picks ascend, so the keyframe before each is found by
            // walking on from the last one looked at.
            while cursor <= pick {
                if table.frames[cursor].1 {
                    keyframe = cursor;
                }
                cursor += 1;
            }
            let pts = table.frames[pick].0;
            match segments.last_mut() {
                Some(segment) if segment_keyframe == Some(keyframe) => {
                    segment.pts.push(pts);
                    segment.decoded_frames = pick - keyframe + 1;
                }
                _ => {
                    if segments.len() == MAX_SEGMENTS {
                        return None;
                    }
                    segment_keyframe = Some(keyframe);
                    segments.push(SeekSegment {
                        seek_seconds: table.seek_seconds(keyframe),
                        pts: vec![pts],
                        decoded_frames: pick - keyframe + 1,
                    });
                }
            }
        }
        Some(Self {
            segments,
            last_pick,
        })
    }

    /// Source frames the plan's runs decode in all.
    pub(crate) fn decoded_frames(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| segment.decoded_frames)
            .sum()
    }

    /// Source frames the sequential decode runs through for the same picks.
    pub(crate) fn sequential_frames(&self) -> usize {
        self.last_pick + 1
    }

    /// The plan's work in source frames, each run's fixed cost counted as
    /// [`RUN_OVERHEAD_FRAMES`].
    pub(crate) fn cost(&self) -> usize {
        self.decoded_frames()
            .saturating_add(self.segments.len().saturating_mul(RUN_OVERHEAD_FRAMES))
    }

    /// `true` when the plan is at most half the work of the sequential
    /// decode: a clear win, not a trade of one cost for another.
    pub(crate) fn pays_off(&self) -> bool {
        self.cost().saturating_mul(2) <= self.sequential_frames()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A listing of `total` frames at 30 fps in a 1/15360 time base with a
    /// keyframe every `gop` frames, in decode order with the B-frame
    /// reordering the encoder of the test clips writes.
    fn listing(total: usize, gop: usize, start_time: &str) -> String {
        let mut lines = vec![format!("format,{start_time}")];
        let mut pts: Vec<usize> = (0..total).collect();
        // Decode order: each keyframe first, then the frames behind it with
        // pairs swapped, as a one-B-frame pattern would be listed.
        for chunk in pts.chunks_mut(gop) {
            let rest = &mut chunk[1..];
            for pair in rest.chunks_mut(2) {
                if pair.len() == 2 {
                    pair.swap(0, 1);
                }
            }
        }
        for index in pts {
            let key = if index % gop == 0 { "K__" } else { "___" };
            lines.push(format!("packet,{},{key}", index * 512));
        }
        lines.push("stream,1/15360".to_string());
        lines.join("\n")
    }

    #[test]
    fn the_listing_is_read_in_presentation_order() {
        let table = FrameTable::parse(listing(600, 48, "0.000000").as_bytes()).expect("a table");
        assert_eq!(table.len(), 600);
        assert_eq!(table.time_base, (1, 15360));
        assert_eq!(table.start_time_us, 0);
        assert!(table.frames.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert_eq!(
            table
                .frames
                .iter()
                .enumerate()
                .filter(|(_, (_, key))| *key)
                .map(|(index, _)| index)
                .collect::<Vec<_>>(),
            (0..600).step_by(48).collect::<Vec<_>>()
        );
        assert_eq!(table.seek_seconds(0), "0.000000");
        assert_eq!(table.seek_seconds(48), "1.600000");
        // Frame 1 sits at 512 / 15360 s = 33333.33 us, rounded to the nearest.
        assert_eq!(table.seek_seconds(1), "0.033333");
    }

    #[test]
    fn the_seek_position_is_measured_from_the_container_start() {
        // The listing's timestamps are absolute: a container that starts at
        // 7 s lists its first frame at 7 s, and the decoder tool wants 0.
        let mut shifted = String::new();
        for line in listing(600, 48, "7.000000").lines() {
            if let Some(rest) = line.strip_prefix("packet,") {
                let (pts, flags) = rest.split_once(',').expect("pts and flags");
                let pts: i64 = pts.parse().expect("ticks");
                shifted.push_str(&format!("packet,{},{flags}\n", pts + 7 * 15360));
            } else {
                shifted.push_str(line);
                shifted.push('\n');
            }
        }
        let table = FrameTable::parse(shifted.as_bytes()).expect("a table");
        assert_eq!(table.seek_seconds(0), "0.000000");
        assert_eq!(table.seek_seconds(48), "1.600000");
        assert_eq!(table.seek_seconds(599), "19.966667");
    }

    #[test]
    fn listings_the_plan_cannot_trust_are_declined() {
        let good = listing(96, 48, "0.000000");
        assert!(FrameTable::parse(good.as_bytes()).is_some());
        // A packet without a timestamp.
        assert!(FrameTable::parse(good.replace("packet,512,", "packet,N/A,").as_bytes()).is_none());
        // A corrupt packet.
        assert!(FrameTable::parse(good.replace("512,___", "512,__C").as_bytes()).is_none());
        // Two frames at one timestamp.
        assert!(
            FrameTable::parse(good.replace("packet,1024,", "packet,512,").as_bytes()).is_none()
        );
        // No time base, or one too fine for a microsecond position.
        assert!(FrameTable::parse(good.replace("stream,1/15360", "").as_bytes()).is_none());
        assert!(
            FrameTable::parse(good.replace("stream,1/15360", "stream,1/600000").as_bytes())
                .is_none()
        );
        // A stream listed twice with one time base is one stream; with two it is declined.
        let twice = format!("{good}\nstream,1/15360");
        assert!(FrameTable::parse(twice.as_bytes()).is_some());
        let disagreeing = format!("{good}\nstream,1/90000");
        assert!(FrameTable::parse(disagreeing.as_bytes()).is_none());
        // A first frame that is no keyframe.
        assert!(
            FrameTable::parse(good.replace("packet,0,K__", "packet,0,___").as_bytes()).is_none()
        );
        // Discarded lead-in packets are not frames; a trailing comma and an
        // unknown start time (as a transport stream lists them) are fine.
        let with_lead_in = format!("packet,-1024,_D_,\npacket,-512,_D_,\n{good}")
            .replace("format,0.000000", "format,N/A");
        let table = FrameTable::parse(with_lead_in.as_bytes()).expect("a table");
        assert_eq!(table.len(), 96);
        assert_eq!(table.start_time_us, 0);
    }

    #[test]
    fn picks_are_grouped_behind_their_keyframes_and_the_work_is_counted() {
        let table = FrameTable::parse(listing(600, 48, "0.000000").as_bytes()).expect("a table");
        let picks = [0, 37, 48, 49, 101, 250, 598, 599];
        let plan = SeekPlan::build(&table, 600, &picks).expect("a plan");
        let segments: Vec<(&str, Vec<i64>, usize)> = plan
            .segments
            .iter()
            .map(|segment| {
                (
                    segment.seek_seconds.as_str(),
                    segment.pts.clone(),
                    segment.decoded_frames,
                )
            })
            .collect();
        assert_eq!(
            segments,
            vec![
                ("0.000000", vec![0, 37 * 512], 38),
                ("1.600000", vec![48 * 512, 49 * 512], 2),
                ("3.200000", vec![101 * 512], 6),
                ("8.000000", vec![250 * 512], 11),
                ("19.200000", vec![598 * 512, 599 * 512], 24),
            ]
        );
        assert_eq!(plan.decoded_frames(), 38 + 2 + 6 + 11 + 24);
        assert_eq!(plan.sequential_frames(), 600);
        assert_eq!(plan.cost(), 81 + 5 * RUN_OVERHEAD_FRAMES);
        // Five runs for a 20-second clip: not worth it; the stream is short.
        assert!(!plan.pays_off());
        assert_eq!(
            plan.segments[1].select_filter(),
            r"select='eq(pts\,24576)+eq(pts\,25088)'"
        );
        // The run ends right after its last pick: a pick the seek missed
        // costs the run its segment, not the rest of the stream.
        assert_eq!(plan.segments[1].end_pts(), 25089);
        assert_eq!(
            plan.segments[1].frame_filter(),
            r"trim=end_pts=25089,select='eq(pts\,24576)+eq(pts\,25088)'"
        );
        assert_eq!(plan.segments[0].end_pts(), 37 * 512 + 1);
    }

    /// Half an hour at 30 fps with a keyframe every 250 frames, sampled by
    /// the loader's rule at a budget of 32: each pick is its own run, the
    /// work is the picks' neighbourhoods, and the plan is a clear win.
    #[test]
    fn a_long_clip_sampled_sparsely_is_a_clear_win() {
        let total = 54_000;
        let table = FrameTable::parse(listing(total, 250, "0.000000").as_bytes()).expect("a table");
        let picks: Vec<usize> = (0..32).map(|i| i * (total - 1) / 31).collect();
        let plan = SeekPlan::build(&table, total, &picks).expect("a plan");
        assert_eq!(plan.segments.len(), 32);
        assert_eq!(plan.sequential_frames(), total);
        let decoded = plan.decoded_frames();
        assert!(decoded <= 32 * 250, "{decoded}");
        assert!(plan.cost() * 2 <= total, "{}", plan.cost());
        assert!(plan.pays_off());
        // The last pick is the last frame, 249 frames behind its keyframe.
        let last = plan.segments.last().expect("a run");
        assert_eq!(last.decoded_frames, 250);
        assert_eq!(last.seek_seconds, "1791.666667");
        assert_eq!(last.pts, vec![(total as i64 - 1) * 512]);
    }

    #[test]
    fn a_plan_that_is_no_clear_win_does_not_pay_off() {
        let table = FrameTable::parse(listing(600, 48, "0.000000").as_bytes()).expect("a table");
        // Every keyframe owns a pick near its end: nearly the whole stream is
        // decoded anyway, plus a run per keyframe.
        let picks: Vec<usize> = (0..600)
            .step_by(48)
            .map(|key| (key + 47).min(599))
            .collect();
        let plan = SeekPlan::build(&table, 600, &picks).expect("a plan");
        assert_eq!(plan.segments.len(), 13);
        assert_eq!(plan.decoded_frames(), 12 * 48 + 24);
        assert!(!plan.pays_off());
        // One pick: the first keyframe's run is the whole plan, and it is
        // only half the cost of the stream when the stream is long enough.
        let plan = SeekPlan::build(&table, 600, &[599]).expect("a plan");
        assert_eq!(plan.decoded_frames(), 24);
        assert!(plan.pays_off());
        let plan = SeekPlan::build(&table, 600, &[100]).expect("a plan");
        assert_eq!(plan.decoded_frames(), 5);
        assert!(!plan.pays_off(), "{} frames of work for 101", plan.cost());
    }

    #[test]
    fn a_table_of_another_stream_makes_no_plan() {
        let table = FrameTable::parse(listing(600, 48, "0.000000").as_bytes()).expect("a table");
        assert!(SeekPlan::build(&table, 601, &[0, 599]).is_none());
        assert!(SeekPlan::build(&table, 600, &[0, 600]).is_none());
        assert!(SeekPlan::build(&table, 600, &[]).is_none());
    }
}
