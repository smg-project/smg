use kv_index::{compute_request_content_hashes, salt::namespace_seed, XXH3_SEED};

use super::{prompt_token_count, HashSeeds, PromptHashes};
use crate::{
    document::{applied::AppliedSpec, spec::PolicySpec},
    engine::view::{HashKind, RequestFacts, TEXT_BYTES_PER_TOKEN},
};

const BLOCK_SIZE: u32 = 16;
/// A text of two windows and a tail, at a block of sixteen tokens.
const RECORDED_TEXT: &str = "A request's text is hashed by windows of its bytes, as deep as \
                                 the generation reads, and a tail shorter than one window gives \
                                 no hash at all.";
const RECORDED_TOKENS: [u64; 2] = [0xea06_fc71_a5ee_6e88, 0x1fbf_39bd_106b_e0b5];
const RECORDED_SCOPED_TOKENS: [u64; 2] = [0x3c6f_83b1_c88e_dc76, 0xcbfd_d2b6_595c_c9c6];
const RECORDED_WINDOWS: [u64; 2] = [0xc67b_5989_9662_3ba3, 0x627b_9253_c3e5_d0f8];

/// The cache-aware preset's document: it reads token blocks and text
/// windows.
fn document() -> AppliedSpec {
    let document = PolicySpec::preset("cache_aware").expect("the preset");
    AppliedSpec::validate(&document).expect("the preset validates")
}

/// The hashes of a prompt under that document, for a model whose block
/// size of record is [`BLOCK_SIZE`].
fn hashed(tokens: Option<&[u32]>, text: Option<&str>, seeds: HashSeeds) -> PromptHashes {
    PromptHashes::under_document(&document(), Some(BLOCK_SIZE), tokens, text, seeds)
}

/// The facts `hashes` writes into a request that holds none.
fn written(hashes: &PromptHashes) -> RequestFacts<'_> {
    let mut facts = RequestFacts::default();
    hashes.write_to(&mut facts);
    facts
}

/// The five facts `hashes` writes, as one value to compare.
fn five(hashes: &PromptHashes) -> (Vec<u64>, HashKind, u32, u64, u32) {
    let facts = written(hashes);
    (
        facts.block_hashes.to_vec(),
        facts.hash_kind,
        facts.block_size,
        facts.namespace_seed,
        facts.prompt_tokens,
    )
}

/// The block hashes of three prompts, as values of record: a plain
/// token prompt, the same tokens under a partition's event seed, and a
/// text prompt. The gateway's case reads the same values through its
/// own cache partition.
#[test]
fn a_prompts_hashes_are_the_recorded_ones() {
    let tokens: Vec<u32> = (1..=40).collect();
    let plain = hashed(Some(&tokens), None, HashSeeds::PLAIN);
    assert_eq!(written(&plain).block_hashes, RECORDED_TOKENS);
    let seeds = HashSeeds {
        event: namespace_seed(None, Some("partition")),
        partition: 7,
    };
    let scoped = hashed(Some(&tokens), None, seeds);
    let facts = written(&scoped);
    assert_eq!(facts.block_hashes, RECORDED_SCOPED_TOKENS);
    assert_eq!(facts.namespace_seed, 7);
    let windows = hashed(None, Some(RECORDED_TEXT), HashSeeds::PLAIN);
    let facts = written(&windows);
    assert_eq!(facts.block_hashes, RECORDED_WINDOWS);
    // A text states no count of tokens. Its length is counted in
    // characters: a stage of this document reads a request's size.
    let chars = RECORDED_TEXT.chars().count() as u32;
    assert_eq!((facts.prompt_tokens, facts.prompt_chars), (0, chars));
}

/// A model whose workers state no block size is hashed by the
/// document's. A size of record wins once there is one, and one of
/// zero states no hash.
#[test]
fn a_model_with_no_block_size_of_record_is_hashed_by_the_documents() {
    let document = document();
    let size = document.block_size();
    let tokens: Vec<u32> = (1..=2 * size).collect();
    for (recorded, blocks, block_size) in [
        (None, 2, size),
        (Some(0), 0, 0),
        (Some(2 * size), 1, 2 * size),
    ] {
        let seeds = HashSeeds::PLAIN;
        let hashes = PromptHashes::under_document(&document, recorded, Some(&tokens), None, seeds);
        let facts = written(&hashes);
        assert_eq!(
            (facts.block_hashes.len(), facts.block_size),
            (blocks, block_size),
            "{recorded:?}"
        );
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
        assert_eq!(facts.namespace_seed, XXH3_SEED);
    }
}

#[test]
fn the_plain_seeds_give_the_index_crates_plain_hashes() {
    let tokens: Vec<u32> = (1..=2 * BLOCK_SIZE).collect();
    let hashes = hashed(Some(&tokens), None, HashSeeds::PLAIN);
    let expected: Vec<_> = compute_request_content_hashes(&tokens, BLOCK_SIZE as usize)
        .into_iter()
        .map(|hash| hash.0)
        .collect();
    let facts = written(&hashes);
    assert_eq!(facts.block_hashes, expected);
    assert_eq!(facts.namespace_seed, XXH3_SEED);
}

#[test]
fn a_tail_shorter_than_a_block_gives_no_hash() {
    let tokens = [1; BLOCK_SIZE as usize + 1];
    let hashes = hashed(Some(&tokens), None, HashSeeds::PLAIN);
    let facts = written(&hashes);
    assert_eq!(facts.block_hashes.len(), 1);
    assert_eq!(facts.prompt_tokens, BLOCK_SIZE + 1);
}

/// A prompt that holds no whole block states one block, the hash of
/// what it has, as the classic's prefix policy keys a short prompt. Two
/// short prompts are two blocks, and an empty prompt states none. The
/// token prompt states its ids' count, and the text its 25 characters.
#[test]
fn a_prompt_under_one_block_states_one_block_of_what_it_has() {
    let tokens = |tokens: &[u32]| hashed(Some(tokens), None, HashSeeds::PLAIN);
    let text = |text: &str| hashed(None, Some(text), HashSeeds::PLAIN);
    let (sevens, eights) = ([7; BLOCK_SIZE as usize - 1], [8; 3]);
    for (short, again, other, block_size, sizes) in [
        (
            tokens(&sevens),
            tokens(&sevens),
            tokens(&eights),
            BLOCK_SIZE,
            (BLOCK_SIZE - 1, 0),
        ),
        (
            text("a prompt under one window"),
            text("a prompt under one window"),
            text("another"),
            BLOCK_SIZE * TEXT_BYTES_PER_TOKEN,
            (0, 25),
        ),
    ] {
        let facts = written(&short);
        assert_eq!(facts.block_hashes.len(), 1, "{short:?}");
        assert_eq!(facts.block_hashes, written(&again).block_hashes);
        assert_ne!(facts.block_hashes, written(&other).block_hashes);
        assert_eq!(facts.block_size, block_size);
        assert_eq!((facts.prompt_tokens, facts.prompt_chars), sizes);
    }
    for empty in [tokens(&[]), text("")] {
        assert!(written(&empty).block_hashes.is_empty(), "{empty:?}");
    }
}

/// The hashes write their five facts into a request that holds no
/// prompt size. A size the request already holds stays, and so does
/// each length of its text it holds.
#[test]
fn a_prompt_size_the_facts_hold_wins_over_the_hashings_count() {
    let seeds = HashSeeds {
        event: XXH3_SEED,
        partition: 9,
    };
    let hashes = hashed(Some(&[3; 88]), None, seeds);
    let facts = written(&hashes);
    assert_eq!(
        (
            facts.hash_kind,
            facts.block_size,
            facts.namespace_seed,
            facts.prompt_tokens
        ),
        (HashKind::Token, BLOCK_SIZE, 9, 88)
    );
    assert!(!facts.block_hashes.is_empty());
    let mut sized = RequestFacts {
        prompt_tokens: 37,
        ..RequestFacts::default()
    };
    hashes.write_to(&mut sized);
    assert_eq!(sized.prompt_tokens, 37);
    assert_eq!(sized.block_hashes, facts.block_hashes);
    let worded = hashed(None, Some("a text of its own"), seeds);
    let mut stated = RequestFacts {
        prompt_chars: 5,
        prompt_bytes: 6,
        ..RequestFacts::default()
    };
    worded.write_to(&mut stated);
    let lengths = (stated.prompt_chars, stated.prompt_bytes);
    assert_eq!((stated.prompt_tokens, lengths), (0, (5, 6)));
}

/// A prompt of text alone states no count of tokens. Its length in
/// bytes is at hand and is stated under any document. Its length in
/// characters is counted only under a document a stage of which reads
/// a request's size: under any other the text is not passed over.
/// Token ids win: a prompt that holds both states its ids' count and no
/// length. The text here is of two-byte characters: a reader that sizes
/// it by the one rule reads its bytes over four, 20 tokens, under
/// either document.
#[test]
fn a_text_alone_states_its_length_in_characters_where_a_stage_reads_a_size() {
    let sized = document();
    let plain = PolicySpec::preset("round_robin").expect("the preset");
    let plain = AppliedSpec::validate(&plain).expect("the preset validates");
    assert!(sized.needs().prompt_size && !plain.needs().prompt_size);
    let text = "\u{e9}".repeat(40);
    assert_eq!(text.len(), 80);
    let sizes = |document: &AppliedSpec, tokens: Option<&[u32]>| {
        let (size, seeds) = (Some(BLOCK_SIZE), HashSeeds::PLAIN);
        let hashes = PromptHashes::under_document(document, size, tokens, Some(&text), seeds);
        let facts = written(&hashes);
        let lengths = (facts.prompt_chars, facts.prompt_bytes);
        (facts.prompt_tokens, lengths, facts.prompt_size())
    };
    assert_eq!(sizes(&sized, None), (0, (40, 80), 20));
    assert_eq!(sizes(&plain, None), (0, (0, 80), 20));
    assert_eq!(sizes(&sized, Some(&[3; 9])), (9, (0, 0), 9));
}

/// No block is hashed past the depth the document reads, of token
/// blocks and of text windows. A document that reads no hashes gets
/// none, and the model's block size is still written.
#[test]
fn no_block_is_hashed_past_the_documents_depth() {
    let spec = document();
    let window = BLOCK_SIZE * TEXT_BYTES_PER_TOKEN;
    let blocks = spec.block_hashes_read(HashKind::Token, BLOCK_SIZE);
    let windows = spec.block_hashes_read(HashKind::Text, window);
    assert!(blocks > 0 && windows > 0);
    let tokens: Vec<u32> = (0..(blocks as u32 + 1) * BLOCK_SIZE).collect();
    let hashes = hashed(Some(&tokens), None, HashSeeds::PLAIN);
    let facts = written(&hashes);
    assert_eq!(facts.block_hashes.len(), blocks);
    assert_eq!(facts.prompt_tokens, tokens.len() as u32);
    let text = "a".repeat((windows + 1) * window as usize);
    let hashes = hashed(None, Some(&text), HashSeeds::PLAIN);
    assert_eq!(written(&hashes).block_hashes.len(), windows);

    let unread = PolicySpec::preset("random").expect("the preset");
    let unread = AppliedSpec::validate(&unread).expect("the preset validates");
    let (size, seeds) = (Some(BLOCK_SIZE), HashSeeds::PLAIN);
    let hashes = PromptHashes::under_document(&unread, size, Some(&tokens), None, seeds);
    let facts = written(&hashes);
    assert!(facts.block_hashes.is_empty());
    assert_eq!(facts.block_size, BLOCK_SIZE);
    assert_eq!(facts.prompt_tokens, tokens.len() as u32);
}

/// A text writes its kind, the size of one window and the partition's
/// seed, which it is hashed under. Another partition moves every window.
/// The event seed moves none: no worker hashes text.
#[test]
fn a_text_is_hashed_under_the_partition_seed_alone() {
    let window = BLOCK_SIZE * TEXT_BYTES_PER_TOKEN;
    let text = "a".repeat(2 * window as usize);
    let windows =
        |event: u64, partition: u64| hashed(None, Some(&text), HashSeeds { event, partition });
    let keyed = windows(5, 11);
    let facts = written(&keyed);
    assert_eq!(
        (facts.hash_kind, facts.block_size, facts.namespace_seed),
        (HashKind::Text, window, 11)
    );
    assert_eq!(facts.block_hashes.len(), 2);
    assert_eq!(written(&windows(6, 11)).block_hashes, facts.block_hashes);
    let other = windows(5, 12);
    assert!(facts
        .block_hashes
        .iter()
        .zip(written(&other).block_hashes)
        .all(|(keyed, other)| keyed != other));
}

/// The hashing itself reads no document: it hashes by the block size and
/// the depth it is handed. The values are the recorded ones.
#[test]
fn the_hashing_alone_reads_the_block_size_and_the_depth_it_is_handed() {
    let tokens: Vec<u32> = (1..=40).collect();
    let whole = PromptHashes::of_tokens(&tokens, BLOCK_SIZE, usize::MAX, HashSeeds::PLAIN);
    assert_eq!(written(&whole).block_hashes, RECORDED_TOKENS);
    let first = PromptHashes::of_tokens(&tokens, BLOCK_SIZE, 1, HashSeeds::PLAIN);
    assert_eq!(written(&first).block_hashes, &RECORDED_TOKENS[..1]);
    let none = PromptHashes::of_tokens(&tokens, BLOCK_SIZE, 0, HashSeeds::PLAIN);
    assert_eq!(
        five(&none),
        (Vec::new(), HashKind::Token, BLOCK_SIZE, XXH3_SEED, 40)
    );
    let window = BLOCK_SIZE * TEXT_BYTES_PER_TOKEN;
    let whole = PromptHashes::of_text(RECORDED_TEXT, window, usize::MAX, false, HashSeeds::PLAIN);
    assert_eq!(written(&whole).block_hashes, RECORDED_WINDOWS);
    let first = PromptHashes::of_text(RECORDED_TEXT, window, 1, false, HashSeeds::PLAIN);
    assert_eq!(written(&first).block_hashes, &RECORDED_WINDOWS[..1]);
}

/// Token ids win over a text, a text is hashed where there are none, and
/// a prompt with neither has no hashes and asks for no block size.
#[test]
fn token_ids_win_over_a_text_and_a_prompt_with_neither_has_no_hashes() {
    let document = document();
    let tokens: Vec<u32> = (1..=40).collect();
    let under = |tokens: Option<&[u32]>, text: Option<&str>| {
        let seeds = HashSeeds::PLAIN;
        PromptHashes::under_document(&document, Some(BLOCK_SIZE), tokens, text, seeds)
    };
    let both = under(Some(&tokens), Some(RECORDED_TEXT));
    assert_eq!(five(&both), five(&under(Some(&tokens), None)));
    assert_eq!(written(&both).block_hashes, RECORDED_TOKENS);
    let text = under(None, Some(RECORDED_TEXT));
    assert_eq!(written(&text).hash_kind, HashKind::Text);
    assert_eq!(written(&text).block_hashes, RECORDED_WINDOWS);
    let plain = PromptHashes {
        namespace_seed: XXH3_SEED,
        ..PromptHashes::default()
    };
    assert_eq!(five(&under(None, None)), five(&plain));
    let neither = PromptHashes::of_prompt(None, None, HashSeeds::PLAIN, |kind| {
        panic!("a prompt with neither asked for the size of a {kind:?} block")
    });
    assert_eq!(five(&neither), five(&plain));
    // Seeds that differ: the partition's is kept, not the event's.
    let parted = HashSeeds {
        event: 3,
        partition: 7,
    };
    let neither = PromptHashes::of_prompt(None, None, parted, |kind| {
        panic!("a prompt with neither asked for the size of a {kind:?} block")
    });
    assert_eq!(written(&neither).namespace_seed, 7);
}

#[test]
fn the_prompt_token_count_saturates_at_u32_max() {
    let past_max = (u32::MAX as usize).saturating_add(1);
    assert_eq!(prompt_token_count(past_max), u32::MAX);
}
