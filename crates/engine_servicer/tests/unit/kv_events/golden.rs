use zeromq::ZmqMessage;

/// `KVEventBatch(ts=1700000000.5, data_parallel_rank=None)` with a
/// `BlockStored` of two sha256-byte hashes (`00..00 80 00..00`,
/// `00..00 ff..fe`), parent 7, tokens 1..=8, block size 4, medium GPU; a
/// `BlockStored` of int hash 0x1234, no parent, tokens [9, 10], block
/// size 2, lora_id 3, group_idx 0, kv_cache_spec_kind full_attention; an
/// unaligned `BlockStored` (one hash, block size 4, three tokens); a
/// `BlockRemoved` of [0x1234, ff..ff]; an `AllBlocksCleared`.
pub(crate) const BATCH1: &str = "93cb41d954fc402000009588a474797065ab426c6f636b53746f726564ac626c6f636b5f68617368657392c4200000000000000000000000000000000000000000000000008000000000000000c420000000000000000000000000000000000000000000000000fffffffffffffffeb1706172656e745f626c6f636b5f6861736807a9746f6b656e5f696473980102030405060708aa626c6f636b5f73697a6504a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c08aa474797065ab426c6f636b53746f726564ac626c6f636b5f68617368657391cd1234b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647392090aaa626c6f636b5f73697a6502a76c6f72615f696403a66d656469756dc0a96c6f72615f6e616d65c0a967726f75705f69647800b26b765f63616368655f737065635f6b696e64ae66756c6c5f617474656e74696f6e88a474797065ab426c6f636b53746f726564ac626c6f636b5f6861736865739105b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647393010203aa626c6f636b5f73697a6504a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c083a474797065ac426c6f636b52656d6f766564ac626c6f636b5f68617368657392cd1234c420ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa66d656469756da347505581a474797065b0416c6c426c6f636b73436c6561726564c0";

/// `KVEventBatch(ts=1700000001.0, data_parallel_rank=1)` with one
/// `BlockStored`: int hash 42, parent 41, tokens [100, 101], block size 2.
pub(crate) const BATCH2: &str = "93cb41d954fc404000009188a474797065ab426c6f636b53746f726564ac626c6f636b5f686173686573912ab1706172656e745f626c6f636b5f6861736829a9746f6b656e5f696473926465aa626c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c001";

pub(crate) fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

/// A publisher message: `[topic, sequence (u64 big-endian), payload]`.
pub(crate) fn frame(topic: &[u8], sequence: u64, payload: &[u8]) -> ZmqMessage {
    let mut message = ZmqMessage::from(topic.to_vec());
    message.push_back(sequence.to_be_bytes().to_vec().into());
    message.push_back(payload.to_vec().into());
    message
}
