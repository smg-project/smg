"""Prepared only: real CPU tensor/method controls, not full SGLang imports."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mrope_delta_cpu", HERE / "mrope_delta.py")
delta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delta)


def request(length=89, value=None):
    return NS(origin_input_ids=list(range(length)), pd_mrope_delta=value,
              pd_mrope_prompt_len=length if value is not None else None)


def method(module, cls, name, patched):
    manifest = json.loads((HERE / "mrope_manifest.json").read_text())
    row = next(r for r in manifest["modules"] if r["module"] == module)
    sources = json.loads(Path(os.environ["EPD_MROPE_ACTUAL_SOURCES"]).read_text())
    raw = Path(sources[module]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["original_sha256"]
    text = raw.decode()
    if patched:
        for before, after in json.loads((HERE / row["patch"]).read_text()):
            assert text.count(before) == 1
            text = text.replace(before, after, 1)
        assert hashlib.sha256(text.encode()).hexdigest() == row["patched_sha256"]
    parent = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    scope = {"torch": torch, "get_exec": lambda: NS(deterministic=NS(rl_on_policy_target=None)),
             "_smg_write_mrope_delta": delta.write, "_smg_pd_mrope_deltas": delta.decode_deltas,
             "_smg_pd_mrope_tail_delta": delta.tail_delta, "_smg_pd_mrope_reject_spec": delta.reject_spec}
    exec(compile(tree, str(sources[module]), "exec"), scope)
    return scope[name]


class Controls(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, SMG_EPD_MROPE_DELTA_HANDOFF="1",
                              SMG_EPD_POSITION_MODEL="Qwen/Qwen3.5-9B")
        self.env.start()
        self.addCleanup(self.env.stop)
        self.assertFalse(torch.cuda.is_initialized())

    def test_codec_off_roundtrip_reuse_and_rejection(self):
        row = torch.arange(16, dtype=torch.int32)
        original = row.clone()
        req = request()
        req.multimodal_inputs = NS(mrope_position_delta=torch.tensor([[-60]]))
        with patch.dict(os.environ, SMG_EPD_MROPE_DELTA_HANDOFF="0"):
            delta.write(row, req)
            self.assertTrue(torch.equal(row, original))
        for value in (-60, -56, 0, 1):
            req.multimodal_inputs.mrope_position_delta.fill_(value)
            delta.write(row, req)
            delta.restore(row, req)
            self.assertEqual(req.pd_mrope_delta, value)
            self.assertTrue(torch.equal(row[:7], original[:7]))
            self.assertTrue(torch.equal(row[9:], original[9:]))
        req.multimodal_inputs = None
        row[4] = 0
        delta.write(row, req)
        delta.restore(row, req)
        self.assertIsNone(req.pd_mrope_delta)
        for tag, value, images in ((delta.TAG + 1, 0, 0), (0, 0, 70), (0, 1, 0)):
            row[7], row[8], row[4] = tag, value, images
            with self.assertRaises(ValueError): delta.restore(row, req)
        for value in (2**31, -(2**31)-1, True):
            with self.assertRaises(ValueError): delta.scalar(value)

    def test_actual_metadata_write_and_original_tuple_shape(self):
        name = "sglang.srt.disaggregation.utils"
        set_buf = method(name, "MetadataBuffers", "set_buf", True)
        get_buf = method(name, "MetadataBuffers", "get_buf", True)
        fields = "output_ids cached_tokens output_token_logprobs_val output_token_logprobs_idx output_top_logprobs_val output_top_logprobs_idx output_topk_p output_topk_index output_hidden_states bootstrap_room".split()
        buffers = NS(**{f: torch.zeros((1,16), dtype=torch.int32) for f in fields},
                     enable_sampling_mask=False, output_dsa_topk_indices=None)
        req = request()
        req.__dict__.update(metadata_buffer_index=0, output_ids=[42], cached_tokens=3,
                            cached_tokens_device=2, cached_tokens_host=1, cached_tokens_storage=0,
                            return_logprob=False, return_sampling_mask=False,
                            hidden_states_tensor=None, bootstrap_room=123)
        req.multimodal_inputs = NS(mrope_position_delta=torch.tensor([[-60]]),
                                  compute_mm_token_counts=lambda: (70,0,0))
        set_buf(buffers, req)
        transferred = get_buf(buffers, 0)
        self.assertEqual(len(transferred), 14)
        self.assertEqual(buffers.cached_tokens[0].nbytes, 64)
        self.assertEqual(transferred[0][0].item(), 42)
        self.assertEqual(transferred[-1][0].item(), 123)
        self.assertEqual(transferred[1][:9].tolist(), [3,2,1,0,70,0,0,delta.TAG,-60])

    def test_actual_decode_old_negative_new_and_mixed(self):
        module = "sglang.srt.model_executor.forward_batch_info"
        old = method(module, "ForwardBatch", "_compute_mrope_positions_decode", False)
        new = method(module, "ForwardBatch", "_compute_mrope_positions_decode", True)
        ret = NS(seq_lens_cpu=torch.tensor([90,89,8]), seq_lens=torch.tensor([90,89,8]))
        batch = NS(reqs=[request(89,-60), request(88,-56), request(7)], multimodal_inputs=[None]*3)
        old(ret, NS(device="cpu"), batch)
        self.assertEqual(ret.mrope_positions.tolist(), [[89,88,7]]*3)
        new(ret, NS(device="cpu"), batch)
        self.assertEqual(ret.mrope_positions.tolist(), [[29,32,7]]*3)
        batch.multimodal_inputs[0] = NS(mrope_positions=None, mrope_position_delta=torch.tensor([[-3]]))
        new(ret, NS(device="cpu"), batch)
        self.assertEqual(ret.mrope_positions.tolist(), [[86,32,7]]*3)

    def test_actual_tail_extend_prompt_and_spec_reject(self):
        module = "sglang.srt.model_executor.forward_batch_info"
        new = method(module, "ForwardBatch", "_compute_mrope_positions_extend", True)
        ret = NS(seq_lens_cpu=torch.tensor([91]))
        batch = NS(reqs=[request(89,-60)], multimodal_inputs=[None], extend_lens=[2], prefix_lens=[89])
        new(ret, NS(device="cpu"), batch)
        self.assertEqual(ret.mrope_positions.tolist(), [[29,30]]*3)
        batch.prefix_lens = [88]
        with self.assertRaises(ValueError): new(ret, NS(device="cpu"), batch)
        with self.assertRaises(ValueError): delta.reject_spec(batch.reqs)
        batch.reqs[0].pd_mrope_prompt_len = 88
        with self.assertRaises(ValueError): delta.decode_deltas(batch.reqs)


if __name__ == "__main__":
    unittest.main()
