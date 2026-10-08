"""Golden msgpack bytes for the Rust KV-event relay tests.

Regenerates the hex payloads in `crates/engine_servicer/src/kv_events.rs`
(`golden` module) from vLLM's own `KVEventBatch` encoder, and prints the
Python relay's conversion of them, which the tests expect byte for byte.
Run it whenever vLLM changes the event layout, with a Python that has vllm,
msgspec and this repo's `smg_grpc_servicer` installed:

    python crates/engine_servicer/scripts/generate_kv_events_golden.py
"""

import msgspec
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)

enc = msgspec.msgpack.Encoder()

# Batch 1: the shapes the Python relay converts (bytes + int hashes, parent,
# lora_id, an unaligned store that must be skipped, a removal, a clear).
batch = KVEventBatch(
    ts=1700000000.5,
    events=[
        BlockStored(
            block_hashes=[
                b"\x00" * 24 + (1 << 63).to_bytes(8, "big"),
                (2**64 - 2).to_bytes(32, "big"),
            ],
            parent_block_hash=7,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            block_size=4,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        ),
        BlockStored(
            block_hashes=[0x1234],
            parent_block_hash=None,
            token_ids=[9, 10],
            block_size=2,
            lora_id=3,
            medium=None,
            lora_name=None,
            group_idx=0,
            kv_cache_spec_kind="full_attention",
        ),
        # 1 hash x block_size 4 != 3 tokens: the relay skips this store but
        # still consumes an event id.
        BlockStored(
            block_hashes=[5],
            parent_block_hash=None,
            token_ids=[1, 2, 3],
            block_size=4,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        ),
        BlockRemoved(block_hashes=[0x1234, b"\xff" * 32], medium="GPU"),
        AllBlocksCleared(),
    ],
    data_parallel_rank=None,
)
payload = enc.encode(batch)
print("BATCH1_HEX", payload.hex())
print("BATCH1_LEN", len(payload))

# Batch 2: dp rank set, one minimal store.
batch2 = KVEventBatch(
    ts=1700000001.0,
    events=[
        BlockStored(
            block_hashes=[42],
            parent_block_hash=41,
            token_ids=[100, 101],
            block_size=2,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
    ],
    data_parallel_rank=1,
)
payload2 = enc.encode(batch2)
print("BATCH2_HEX", payload2.hex())

# Human-readable structure via msgspec's untyped decoder.
dec = msgspec.msgpack.Decoder()
print("BATCH1_STRUCT", dec.decode(payload))
print("BATCH2_STRUCT", dec.decode(payload2))

# The Python relay's conversion of batch 1, for the expected proto values.
from smg_grpc_servicer.kv_events import convert_batch  # noqa: E402

typed = msgspec.msgpack.Decoder(KVEventBatch).decode(payload)
proto, next_id = convert_batch(typed, 9, 0)
print("BATCH1_PROTO", str(proto).replace("\n", " "))
print("BATCH1_NEXT_EVENT_ID", next_id)
typed2 = msgspec.msgpack.Decoder(KVEventBatch).decode(payload2)
proto2, next_id2 = convert_batch(typed2, 10, next_id)
print("BATCH2_PROTO", str(proto2).replace("\n", " "))
print("BATCH2_NEXT_EVENT_ID", next_id2)

# Publisher frame facts.
from vllm.distributed.kv_events import ZmqEventPublisher  # noqa: E402

print("END_SEQ", ZmqEventPublisher.END_SEQ.hex())
print("SEQ5", (5).to_bytes(8, "big").hex())
