use super::*;
use crate::test_support::counting_alloc;

#[test]
fn test_parse_sse_block_event_and_data() {
    let (event, data) =
        parse_sse_block("event: response.created\ndata: {\"type\":\"response.created\"}");
    assert_eq!(event, Some("response.created"));
    assert_eq!(data, "{\"type\":\"response.created\"}");
}

#[test]
fn test_parse_sse_block_data_only() {
    let (event, data) = parse_sse_block("data: {\"type\":\"response.output_text.delta\"}");
    assert_eq!(event, None);
    assert_eq!(data, "{\"type\":\"response.output_text.delta\"}");
}

#[test]
fn test_parse_sse_block_multiline_data() {
    let (event, data) = parse_sse_block("event: x\ndata: line1\ndata: line2");
    assert_eq!(event, Some("x"));
    assert_eq!(data, "line1\nline2");
}

#[test]
fn test_parse_sse_block_control_only_has_empty_data() {
    // A block with no `data:` line yields empty data; all callers skip it.
    let (_event, data) = parse_sse_block(": keep-alive");
    assert!(data.is_empty());

    let (_event, data) = parse_sse_block("event: ping");
    assert!(data.is_empty());
}

/// Feed `bytes` split at every position in `cuts` and collect all blocks
/// plus whatever remains at end of stream.
fn run_chunks(bytes: &[u8], cuts: &[usize]) -> (Vec<String>, String) {
    let mut cp = ChunkProcessor::new();
    let mut blocks = Vec::new();
    let mut start = 0;
    for &cut in cuts.iter().chain(std::iter::once(&bytes.len())) {
        cp.push_chunk(&bytes[start..cut]);
        while let Some(b) = cp.next_block() {
            blocks.push(b.to_string());
        }
        start = cut;
    }
    let rest = if cp.has_remaining() {
        cp.take_remaining()
    } else {
        String::new()
    };
    (blocks, rest)
}

#[test]
fn test_chunk_processor_multibyte_split_at_every_byte() {
    // "化" is 3 bytes and "😀" is 4: split the stream at every possible
    // position, the decoded block must never contain U+FFFD.
    let text = "热迁移是虚拟化的标志😀能力";
    let event = format!("event: response.output_text.done\ndata: {{\"text\":\"{text}\"}}\n\n");
    let bytes = event.as_bytes();
    for cut in 1..bytes.len() {
        let (blocks, rest) = run_chunks(bytes, &[cut]);
        assert_eq!(blocks.len(), 1, "cut at {cut}");
        assert!(
            !blocks[0].contains('\u{FFFD}'),
            "cut at {cut}: {}",
            blocks[0]
        );
        assert!(blocks[0].contains(text), "cut at {cut}");
        assert!(rest.is_empty(), "cut at {cut}");
    }
}

#[test]
fn test_chunk_processor_one_byte_chunks() {
    let event = "data: {\"delta\":\"云计算架构\"}\n\ndata: {\"text\":\"、与私有子网（\"}\n\n";
    let bytes = event.as_bytes();
    let cuts: Vec<usize> = (1..bytes.len()).collect();
    let (blocks, rest) = run_chunks(bytes, &cuts);
    assert_eq!(
        blocks,
        vec![
            "data: {\"delta\":\"云计算架构\"}".to_string(),
            "data: {\"text\":\"、与私有子网（\"}".to_string()
        ]
    );
    assert!(rest.is_empty());
}

#[test]
fn test_chunk_processor_crlf_split_across_chunks() {
    let bytes = b"event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\n";
    for cut in 1..bytes.len() {
        let (blocks, rest) = run_chunks(bytes, &[cut]);
        assert_eq!(
            blocks,
            vec![
                "event: a\ndata: 1".to_string(),
                "event: b\ndata: 2".to_string()
            ],
            "cut at {cut}"
        );
        assert!(rest.is_empty(), "cut at {cut}");
    }
}

#[test]
fn test_chunk_processor_invalid_bytes_still_replaced() {
    // A byte that can never start a UTF-8 sequence is replaced, the rest kept.
    let (blocks, _) = run_chunks(b"data: a\xffb\n\n", &[]);
    assert_eq!(blocks, vec!["data: a\u{FFFD}b".to_string()]);
}

#[test]
fn test_chunk_processor_truncated_tail_at_end_of_stream() {
    // Stream ends in the middle of a character: the remainder is still returned.
    let bytes = "data: 化".as_bytes();
    let (blocks, rest) = run_chunks(&bytes[..bytes.len() - 1], &[]);
    assert!(blocks.is_empty());
    assert_eq!(rest, "data: \u{FFFD}");
}

#[test]
fn test_parse_sse_block_empty_input() {
    let (event, data) = parse_sse_block("");
    assert_eq!(event, None);
    assert!(data.is_empty());
}

/// Blocks handed out are dropped from the buffer at the end of the
/// stream too: `take_remaining` returns the unread tail alone (what the
/// record stores), not the blocks already relayed.
#[test]
fn test_chunk_processor_take_remaining_skips_handed_out_blocks() {
    let (blocks, rest) = run_chunks(b"data: 1\n\ndata: 2", &[]);
    assert_eq!(blocks, vec!["data: 1".to_string()]);
    assert_eq!(rest, "data: 2");
}

/// The blocks handed out from one chunk are dropped when the next chunk
/// arrives, and a block split across the two chunks comes out whole.
#[test]
fn a_block_split_across_chunks_after_handed_out_blocks_comes_out_whole() {
    let mut cp = ChunkProcessor::new();
    cp.push_chunk(b"event: a\ndata: 1\n\nevent: b\ndata: 2");
    assert_eq!(cp.next_block(), Some("event: a\ndata: 1"));
    assert_eq!(cp.next_block(), None);
    cp.push_chunk(b"\n\nevent: c\ndata: 3\n\n");
    assert_eq!(cp.pending, "event: b\ndata: 2\n\nevent: c\ndata: 3\n\n");
    assert_eq!(cp.next_block(), Some("event: b\ndata: 2"));
    assert_eq!(cp.next_block(), Some("event: c\ndata: 3"));
    assert_eq!(cp.next_block(), None);
    assert!(!cp.has_remaining());
    assert_eq!(cp.take_remaining(), "");
}

/// Draining one pushed chunk of ten blocks hands out ten borrowed blocks
/// and allocates nothing: no copy per block, no move of the rest per
/// block (the buffer is compacted once, at the next chunk). The count is
/// printed for the Numbers table.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
fn draining_a_chunk_of_ten_blocks_allocates_nothing() {
    let chunk: String = (0..10)
        .map(|i| {
            format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"sequence_number\":{i},\"delta\":\"a dozen words of answer text that a model streams per step \"}}\n\n"
            )
        })
        .collect();
    let mut cp = ChunkProcessor::new();
    cp.push_chunk(chunk.as_bytes());

    let before = counting_alloc::calls();
    let mut blocks = 0;
    let mut bytes = 0;
    while let Some(block) = cp.next_block() {
        blocks += 1;
        bytes += block.len();
    }
    let calls = counting_alloc::calls() - before;

    eprintln!("draining a chunk of ten blocks: {calls} allocation calls");
    assert_eq!(blocks, 10);
    assert_eq!(
        bytes,
        chunk.len() - 10 * 2,
        "every block, without its blank line"
    );
    assert!(!cp.has_remaining());
    assert_eq!(
        calls, 0,
        "allocation calls while draining a chunk of ten blocks"
    );
}
