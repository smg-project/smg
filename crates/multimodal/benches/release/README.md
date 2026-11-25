# The release comparison

One release of the crate against another, cell by cell: every image processor
configuration by every standard image class on one core, with the time and the CPU per
image, whether the two releases write the same bytes and, where the reference library
holds a class for the processor, the reference's time and bytes on the same files. The
audio front end is read the same way over uncompressed WAV clips.

One source of the harness, the crate's `mm_baseline` example
(`crates/multimodal/examples/mm_baseline.rs`), is compiled against both revisions of the
library, so the two sides differ by the library and by the compiler each release pins,
never by the harness. The recorded run, 1.16.0 against 1.15.0, is in
[RESULTS.md](RESULTS.md) and [audio/RESULTS.md](audio/RESULTS.md).

## Reproduce

Run from the repository root, on Linux: the runner pins with `taskset` and reads
`/proc/stat`. `run.py` and `make_audio.py` use only the Python standard library;
`make_images.py` needs numpy and Pillow, and the reference arm needs the pins of
`scripts/bench/multimodal/requirements.txt`. Each revision builds with the toolchain its
own `rust-toolchain.toml` names.

```sh
B=crates/multimodal/benches/release
python3 $B/run.py build --base v1.15.0 --head HEAD --out run
python3 $B/make_images.py --out run/images
python3 $B/run.py classes --files run/images --configs configs.json --out run/classes
python3 $B/run.py rounds --out run --classes run/classes --cores 2,4,6,8 --rounds 15
python3 $B/run.py summarize --out run --classes run/classes --base-name 1.15.0 --head-name 1.16.0
```

`build` exports each revision into a temporary directory under `run/trees/` with
`git archive` and moves it into place only after the export finishes, so no checkout is
touched and an interrupted export is not reused. It writes the example of `--harness`
(`HEAD` unless named) over the example of both,
gives a revision whose crate manifest has no allocator line the one of the harness
revision, and builds the example in each export. `--cargo-target`, or
`CARGO_TARGET_DIR`, names one target directory for both sides.

`configs.json` names the processor configurations to run. No such file is checked in: a
configuration is a checkpoint's own files, read from wherever they were downloaded to.

```json
{
  "configurations": [
    {
      "name": "qwen3_vl",
      "label": "`qwen3_vl`",
      "model_id": "an id the registry routes to the family",
      "config_dir": "the directory holding the checkpoint's config.json and preprocessor configs"
    }
  ]
}
```

`name` names the class list and the `processor` column of the csv files; `label` is what
the tables print. A second configuration of a family (another checkpoint, another model
type) is a second entry. [environment.json](environment.json) records the SHA-256 of
every config file the recorded run read.

`rounds` runs one harness process at a time on each core of `--cores`; give it cores
nothing else runs on. It starts only when the run directory has no earlier window below
`rounds/`, `windows.log` record or `rounds.json` settings file. Otherwise it refuses before
writing anything and names the first file it found; start an interrupted run in a fresh
directory. Build outputs and a `STOP` file may already be present. A file named `STOP` in
the run directory ends it after the windows in flight. `rounds.json` records the requested
settings so `summarize` can copy them into `environment.json`; `summarize` reads whatever
the run directory holds and can be run at any point.

The harness loads libjpeg-turbo and libwebp at runtime where it finds them and says in
its first output line which it found; `environment.json` carries that line for both
sides. The recorded run found both.

The `harness_blob` key is the Git blob ID of the harness source bytes compiled into both sides.

The other arms write into the same run directory, and `summarize` adds their columns. Each
refuses to start when its own output is already there (`parity/` or `parity.log`,
`reference/` or `reference.log`); to rerun one, remove those and leave the rounds in place:

```sh
# the same bytes on the wider list: 59 classes, two wires, two alpha policies; nothing is timed
python3 $B/run.py classes --standard $B/classes_wide.json --files run/images \
    --configs configs.json --out run/classes-wide
python3 $B/run.py parity --out run --classes run/classes-wide --cores 2,4,6,8

# the reference processors on the same class lists
python3 $B/run.py reference --out run --classes run/classes --cores 2,4,6,8 \
    --python <the interpreter with the pins>
```

The audio front end takes a run directory of its own and the executables already built:

```sh
python3 $B/make_audio.py --out audio-run/clips
python3 $B/run.py classes --standard $B/audio_classes.json --files audio-run/clips \
    --configs audio-configs.json --out audio-run/classes
python3 $B/run.py rounds --out audio-run --classes audio-run/classes --cores 2,4 --rounds 15 \
    --base-bin run/bin/mm_baseline-base --head-bin run/bin/mm_baseline-head
# the build record of those executables: summarize names the two revisions from it
cp run/sides.json audio-run/
python3 $B/run.py summarize --out audio-run --classes audio-run/classes \
    --base-name 1.15.0 --head-name 1.16.0
```

## Measurement contract

- **The cell.** One processor configuration by one class. The time per image is the
  wall time of the class's timed loop over the images it processed: the header read and
  the decode, the processor's one-pass write over the decoded pixels in the dtype of the
  wire, and the move of the wire bytes on a second thread, as the gateway does it. The
  CPU per image is the process CPU time of the same loop, every thread counted. One
  untimed pass and one warm-up come before 10 timed repeats.
- **Settings.** One harness thread and a preprocessing pool of one thread
  (`SMG_MM_THREADS=1`), the bf16 wire, transparency composited over white, the
  gateway's allocator with the gateway's options, release builds.
- **One process per configuration.** A window is one harness process that runs the 20
  classes of one configuration in the order of `classes.json`. The crate's buffer pool
  is process-wide, so a class is read with the pool the classes before it left behind;
  both sides run the same order. A check read two configurations one class a process,
  three rounds (`one_class_check.csv`): the head's time per image was within 5 % of the
  recorded one in 31 of the 40 cells (from 9 % under it to 16 % over), the change
  between the releases moved by under 2 points in 25 cells and by 12 points in the worst,
  two cells crossed from level to faster and one (`qwen3_vl`, `tiff`) from faster to slower.
- **Rounds.** Every round runs both sides of every configuration, the base first in odd
  rounds and the head first in even ones, each process pinned to one core. A cell's time
  is the median over the rounds; its change is the head's median over the base's, minus
  one. A change within 2 % either way reads as level. For a cell that reads slower the
  results also say in how many rounds the head was the slower side: a cell slower in
  every round is a cost, a cell with rounds on both sides of level is not shown to be one.
- **Void windows.** A window is void, and taken again (at most three attempts), when the
  core's other logical CPU was busy for more than 10 % of the window; a third busy window
  in a row is kept and logged `LOUD`. A core without an SMT sibling is never checked.
- **Same bytes.** Two outputs are the same when the shape, the grid and the SHA-256 of
  the decoded pixels, of the float32 write, of the uint8 write and of the wire bytes are
  equal, and every round of a side wrote the same. The wider list compares 59 classes on
  two wires (bf16 and float32) under two alpha policies (composited over white and
  kept): 236 outputs a configuration. A class both sides refuse lies outside the
  processor's own bounds.
- **The reference arm.** `scripts/bench/multimodal/reference.py` loads the image
  processor class the pinned `transformers` holds for the checkpoint's config and times
  its decode and its preprocess with one compute thread, three rounds of 10 repeats.
  The crate's figure beside it also carries the write of the wire bytes. The arm never
  runs code shipped inside a checkpoint: a configuration whose processor is such code
  has no reference column, and neither has one the library holds no class for.
- **Two compilers.** Each release is built by the compiler it pins, so a difference
  between the sides carries the compiler's part too. `environment.json` names both.
- **What it is not.** One thread on one machine: no cell under load, none through the
  gateway, no engine and no network. The images are one seeded synthetic scene in many
  containers, not photographs. Compressed audio and video are not in the recorded run.

## Files

| file | what it is |
|---|---|
| `run.py` | the runner: `build`, `classes`, `rounds`, `parity`, `reference`, `summarize` |
| `make_images.py` | writes the 59 image files of the class lists from one seeded scene |
| `make_audio.py` | writes the 19 WAV clips |
| `classes.json` | the 20 standard classes: 19 image classes and one batch of ten |
| `classes_wide.json` | the 59 classes of the wider list |
| `audio_classes.json` | the 19 WAV classes |
| `images.json`, `audio.json` | name, size and SHA-256 of every file the recorded run read |
| `RESULTS.md` | the recorded image run: the tables |
| `cells.csv` | every cell: both times, the change, the CPU, the bytes, every round's ratio, the reference |
| `rounds.csv.gz` | every window of every round: the time, the CPU, the stage medians (the `rounds.csv` that `summarize` writes, compressed) |
| `environment.json` | commits, compilers, the SHA-256 of the executables, of the class files and of the config files, the reference versions |
| `one_class_check.csv` | the windows of the check that read two configurations one class a process |
| `audio/` | the same four files for the recorded audio run, its `rounds.csv` uncompressed |

## Test data

No media file is checked in. The two makers write every file the three class lists name,
and the same bytes each time under the pinned Pillow and numpy: a maker writes a listing
beside its files, and that listing equals `images.json` or `audio.json` here when the
files are the ones the recorded run read.

Recorded media (photographs, documents, screenshots, clips) lives in the fixture
repository, the one `.github/versions/bellwether.ref` pins, under its `corpus/` tree:
`images/<set>/<name>`, `videos/<set>/<name>`, `audios/<set>/<name>`. The baseline of
`scripts/bench/multimodal` names its classes against that tree, and a class list for this
runner takes those files the same way: a class is a name, a kind and a list of files.

## Verification

```sh
ruff check crates/multimodal/benches/release
ruff format --check crates/multimodal/benches/release
python3 crates/multimodal/benches/release/make_audio.py --out clips
diff clips/audio.json crates/multimodal/benches/release/audio.json
```

`summarize` lists what it could not read under `problems` in `environment.json`; the
recorded run has none.
