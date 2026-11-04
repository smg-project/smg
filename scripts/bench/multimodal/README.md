# The multimodal baseline

One command runs SMG's media pipeline and the serving engine's own media path over the
same files, class by class, one item at a time and under load, and writes the table of
record for the box it runs on: the architecture, the core count and load, the decoders
the crate found at runtime (TurboJPEG, libwebp), the vector-kernel paths of the build,
the preprocessing pool, and per class the stage latencies, the throughput, the CPU per
item, the output shape and the parity of the decoded pixels and the float32 output
against the reference processor.

```
python3 scripts/bench/multimodal/mm_baseline.py \
    --corpus <corpus dir> \
    --config-dir qwen3_vl=<Qwen3-VL checkpoint configs> --config-dir qwen3_asr=<Qwen3-ASR checkpoint configs> \
    --python <interpreter with requirements.txt> --out <run dir> --threads 16 --repeats 10
```

| file | what it is |
|---|---|
| `mm_baseline.py` | the entry point: resolves the classes, materializes the derived files, runs the arms, writes `table.md` and `table.json` |
| `classes.json` | the classes: a file of the fixture repository's corpus per class, or a recipe that derives one from a corpus file where no public file of that shape exists yet |
| `../../../crates/multimodal/examples/mm_baseline.rs` | the crate arm: decode, the gateway's frame clone, the one-pass preprocess into the wire's dtype and the move of the wire bytes, per class, on one or many threads, with digests |
| `reference.py` | the reference arm: the engine's image loader rules and the pinned `transformers` processors on the same files, one process or many with one compute thread each |
| `requirements.txt` | the reference arm's pins (the engine image's own) |

The corpus directory follows the fixture repository's layout (`images/<set>/<name>`,
`videos/<set>/<name>`, `audios/<set>/<name>`); the manifest names every file by set and
name, so a checkout of that repository with the sets pulled serves as is, and so does a
directory holding only the files the classes name. The checkpoint config directories hold
`config.json` and the preprocessor configs (the video one where the checkpoint ships it);
a gateway run needs the tokenizer files beside them.

What a run compares: the reference processor is the one the engine's API server runs
(`transformers` at the engine image's pin, the loader rules of its image path: EXIF
orientation applied, transparency composited over white). The crate's float32 write of a
class equals the reference's float32 output byte for byte when the pipeline is at parity;
the table says `same` or `DIFF` per class for the decoded RGB bytes and the float32 output.
The reference's video arm picks frames the way the engine's loader does for the Qwen3-VL
family and decodes them with OpenCV; its audio arm decodes with `soundfile` and runs the
checkpoint's feature extractor with the engine processor's own arguments.
