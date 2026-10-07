//! Shared by the integration tests: the ways an output is cut into deltas, feeding a parser, and
//! the output bytes each event accounts for.

use symphony::{EngineFinish, Event, Events, Input, ParseError, Parser};

/// The cut offsets to replay `text` with: whole, every two-way split, byte by byte, and thirty
/// seeded plans of one to eight characters; every cut on a character boundary.
pub fn chunkings(text: &str) -> Vec<Vec<usize>> {
    let chars: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let mut plans = vec![vec![]];
    plans.extend(chars.iter().skip(1).map(|&cut| vec![cut]));
    plans.push(chars.iter().skip(1).copied().collect());
    for seed in 1..=30u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut cuts = Vec::new();
        let mut at = 0;
        while at < chars.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            at += 1 + (state >> 33) as usize % 8;
            if at < chars.len() {
                cuts.push(chars[at]);
            }
        }
        plans.push(cuts);
    }
    plans
}

/// The output bytes `event` accounts for.
pub fn bytes_of(event: &Event) -> &str {
    match event {
        Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.text.as_str(),
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
    }
}

/// An uncounted delta of `text`.
pub fn delta(text: &str) -> Input<'_> {
    Input::Delta {
        token_ids: &[],
        text,
        spans: &[],
    }
}

/// An empty prompt.
pub fn prompt() -> Input<'static> {
    Input::Prompt {
        token_ids: &[],
        text: "",
    }
}

/// Feed `text` to `parser` cut at `cuts`, after an empty prompt, then the end with the engine's
/// `finish`, and return the events. With `empty_between`, an empty delta comes before every piece
/// and after the last.
pub fn replay(
    parser: &mut dyn Parser,
    text: &str,
    cuts: &[usize],
    finish: &EngineFinish,
    empty_between: bool,
) -> Result<Vec<Event>, ParseError> {
    replay_after(parser, "", text, cuts, finish, empty_between)
}

/// [`replay`] after a prompt whose text is `prompt_text`: the tail a template leaves before the
/// output, such as Qwen 3.5's `<think>\n`.
pub fn replay_after(
    parser: &mut dyn Parser,
    prompt_text: &str,
    text: &str,
    cuts: &[usize],
    finish: &EngineFinish,
    empty_between: bool,
) -> Result<Vec<Event>, ParseError> {
    let mut out = Events::new();
    parser.feed(
        Input::Prompt {
            token_ids: &[],
            text: prompt_text,
        },
        &mut out,
    )?;
    let mut from = 0;
    for &cut in cuts.iter().chain(std::iter::once(&text.len())) {
        if cut > from {
            if empty_between {
                parser.feed(delta(""), &mut out)?;
            }
            parser.feed(delta(&text[from..cut]), &mut out)?;
            from = cut;
        }
    }
    if empty_between {
        parser.feed(delta(""), &mut out)?;
    }
    parser.feed(
        Input::End {
            finish: finish.clone(),
        },
        &mut out,
    )?;
    Ok(out.drain())
}
