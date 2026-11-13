"""Native layout parity checks; requires the supported community SGLang installation."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang.srt.managers.req_types")
from sglang.srt.managers.req_types import (  # noqa: E402
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.modality import Modality  # noqa: E402
from smg_grpc_servicer.sglang.managed import native  # noqa: E402


def test_native_epd_metadata_roundtrip_without_media_or_embeddings():
    """Preserve native absence as well as identity, sentinel sequences and positions."""
    m = native._proto()
    b = m.InputBinding(prefill_rid="p", layout_contract=native.LAYOUT)
    item = MultimodalDataItem(
        modality=Modality.IMAGE, offsets=[(1, 4)], precomputed_embeddings=torch.zeros((4, 8))
    )
    item.set_pad_value()
    mm = MultimodalInputs.from_processor_output(
        MultimodalProcessorOutput(
            mm_items=[item],
            input_ids=[1, 151655, 151655, 151655, 151655, 2],
            im_token_id=151655,
            mrope_positions=torch.zeros((3, 6), dtype=torch.int64),
            mrope_position_delta=torch.tensor([0]),
        )
    )
    req = SimpleNamespace(
        multimodal_inputs=mm,
        to_finish=None,
        rid="p",
        origin_input_ids_unpadded=[1, 151655, 151655, 151655, 151655, 2],
        origin_input_ids=[1, *item.padding_sequence(4), 2],
    )
    export = native.export_layout(req, b)
    restored = native.decode_input(export, b)
    assert restored.mm_items[0].model_specific_data == {}
    assert restored.mm_items[0].processor_fingerprint is None
    assert restored.mm_items[0].identity == item.identity
    assert restored.mm_items[0].padding_sequence(4) == item.padding_sequence(4)
    assert (
        restored.mm_items[0].feature is None and restored.mm_items[0].precomputed_embeddings is None
    )
    assert torch.equal(restored.mrope_positions, mm.mrope_positions)
