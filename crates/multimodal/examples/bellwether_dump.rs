//! SMG's media pipeline over a list of image files, one JSON line per image, for comparison with
//! bellwether's recorded multimodal fixtures.
//!
//! bellwether (smg-project/bellwether) records, per processor family, what the reference
//! implementation's own image path gives for each image of its corpus: the decoded RGB bytes,
//! the u8 patches, the bf16 and the f32 pixel tensors, each as a SHA-256. This example gives the
//! same for SMG: each listed file decoded the way the gateway decodes inline image bytes (EXIF
//! orientation applied, no size cap), the family's processor built from its
//! `preprocessor_config.json`, the pixel tensor written as the gateway writes it on the wire, in
//! bf16 and in f32, and the digests of all four, so that a recorded line and SMG's line for an
//! image compare field by field. Comparing a whole fixture set is a test's business; by hand,
//! `diff` two dumps, or a dump and the recorded lines of the same images.
//!
//! Settings come from flags, or from environment variables of the same meaning, so a runner can
//! set them without a command line (a flag wins over its variable):
//!
//! | flag | variable | meaning |
//! |---|---|---|
//! | `--list <file>` | `SMG_DUMP_LIST` | the image files, one path per line |
//! | `--out <path>` | `SMG_DUMP_OUT` | in `hash` mode the JSON Lines file written; in `raw` mode the directory the artifacts go to |
//! | `--processor <name>` | `SMG_DUMP_PROCESSOR` | `qwen2_vl`, `qwen3_vl`, `minimax_m3` or `muse_spark`: the processor family |
//! | `--config <file>` | `SMG_DUMP_CONFIG` | the family's `preprocessor_config.json` |
//! | `--mode <hash\|raw>` | `SMG_DUMP_MODE` | `hash` unless given |
//!
//! `hash` writes one JSON object per image: `name` (the file name, the key a recorded line and a
//! dump line share), `path` (as listed), `w` and `h` (the decoded size), `alpha` (whether the
//! file carried transparency), `grid` (the image grid the processor reports, as `t, h, w`,
//! empty where it reports none), `shape` (the pixel tensor's shape), `rgb_sha256` (the
//! decoded RGB bytes), `u8_sha256` (the un-normalized u8 patches; `null` when the processor
//! produces floats), `bf16_sha256` and `f32_sha256` (the wire bytes). An image that cannot
//! be decoded or processed gets `name`, `path` and `error` instead, and the run goes on.
//! `raw` writes the artifacts themselves under the image's name, `<name>.rgb.raw`,
//! `<name>.u8.raw` (when there are u8 patches), `<name>.bf16.raw` and `<name>.f32.raw`; where
//! any two listed files share a name, every file is written under
//! `<position in the list>-<name>` instead, which keeps the files of one list apart whatever
//! the names. `meta.json` carries the processor's name and the per-image lines without the
//! digests, each with the stem its files were written under as `artifact`, for the first
//! differing element.
//!
//! The exit status is 0 once every listed image has its line, errors included, and the last
//! line on standard error is then `DUMP_DONE mode=<mode> processor=<name> ok=<n> err=<n>`. A
//! fault in the setup (an unreadable list or config, an unknown processor, an output that
//! cannot be written) ends the run with status 1 and the fault, as written, on standard error.
//! The crate's preprocessing pool is sized by `SMG_MM_THREADS` as everywhere else; a runner
//! that shards a list across processes sets it to 1.
//!
//! ```text
//! cargo run --release -p llm-multimodal --example bellwether_dump -- \
//!     --list corpus-50.txt --processor qwen3_vl --config preprocessor_config.json --out dump.jsonl
//! ```

use std::{
    collections::HashSet,
    env,
    error::Error,
    fmt::Write as _,
    fs::{self, File},
    io::{BufWriter, Cursor, Write},
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    process::ExitCode,
};

use image::{DynamicImage, ImageReader};
use llm_multimodal::{
    encoder_inputs::{EncoderDtype, EncoderInput, ModelSpecificValue},
    types::{ImageDetail, ImageFrame, ImageSize, ImageSource},
    vision::{
        processors::{
            MiniMaxM3VisionProcessor, MuseSparkVisionProcessor, Qwen2VLProcessor, Qwen3VLProcessor,
        },
        PreProcessorConfig, VisionPreProcessor,
    },
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// A fault in the run's setup, reported on standard error with a non-zero exit status.
type Fault = Box<dyn Error>;

const PROCESSORS: &str = "qwen2_vl, qwen3_vl, minimax_m3, muse_spark";

const USAGE: &str = "bellwether_dump: SMG's media pipeline over a list of image files, one JSON line per image\n\
    \n\
    Options (each has an environment variable of the same meaning; the flag wins):\n\
    \x20 --list <file>        SMG_DUMP_LIST       the image files, one path per line\n\
    \x20 --out <path>         SMG_DUMP_OUT        hash: the JSON Lines file; raw: the output directory\n\
    \x20 --processor <name>   SMG_DUMP_PROCESSOR  qwen2_vl, qwen3_vl, minimax_m3 or muse_spark\n\
    \x20 --config <file>      SMG_DUMP_CONFIG     the family's preprocessor_config.json\n\
    \x20 --mode <hash|raw>    SMG_DUMP_MODE       hash unless given\n\
    \x20 --help               this text";

/// What the run writes: digests per image to one file, or the artifacts into a directory.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Hash,
    Raw,
}

impl Mode {
    fn parse(name: &str) -> Result<Self, Fault> {
        match name {
            "hash" => Ok(Self::Hash),
            "raw" => Ok(Self::Raw),
            other => Err(format!("mode {other:?} is neither hash nor raw").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Hash => "hash",
            Self::Raw => "raw",
        }
    }
}

/// The run's settings, from the flags over the environment variables.
struct Options {
    list: PathBuf,
    out: PathBuf,
    processor: String,
    config: PathBuf,
    mode: Mode,
}

impl Options {
    /// `None` when `--help` was asked for and printed.
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Option<Self>, Fault> {
        let mut given: [Option<String>; 5] = Default::default();
        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                usage();
                return Ok(None);
            }
            let slot = match arg.as_str() {
                "--list" => 0,
                "--out" => 1,
                "--processor" => 2,
                "--config" => 3,
                "--mode" => 4,
                other => return Err(format!("unknown argument {other:?}\n{USAGE}").into()),
            };
            let value = args
                .next()
                .ok_or_else(|| format!("{arg} takes a value\n{USAGE}"))?;
            given[slot] = Some(value);
        }
        let setting = |slot: usize, variable: &str, flag: &str| -> Result<String, Fault> {
            given[slot]
                .clone()
                .or_else(|| env::var(variable).ok())
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    format!("{variable} is not set and {flag} was not given\n{USAGE}").into()
                })
        };
        let mode = match given[4].clone().or_else(|| env::var("SMG_DUMP_MODE").ok()) {
            Some(name) if !name.is_empty() => Mode::parse(&name)?,
            _ => Mode::Hash,
        };
        Ok(Some(Self {
            list: PathBuf::from(setting(0, "SMG_DUMP_LIST", "--list")?),
            out: PathBuf::from(setting(1, "SMG_DUMP_OUT", "--out")?),
            processor: setting(2, "SMG_DUMP_PROCESSOR", "--processor")?,
            config: PathBuf::from(setting(3, "SMG_DUMP_CONFIG", "--config")?),
            mode,
        }))
    }
}

#[expect(clippy::print_stdout, reason = "the usage text is what --help prints")]
fn usage() {
    println!("{USAGE}");
}

/// The family's processor, built from its preprocessor config as the gateway builds it.
fn build_processor(
    name: &str,
    config: &PreProcessorConfig,
) -> Result<Box<dyn VisionPreProcessor>, Fault> {
    Ok(match name {
        "qwen2_vl" => Box::new(Qwen2VLProcessor::from_preprocessor_config(config)),
        "qwen3_vl" => Box::new(Qwen3VLProcessor::from_preprocessor_config(config)),
        "minimax_m3" => Box::new(MiniMaxM3VisionProcessor::from_preprocessor_config(config)?),
        // Reads its settings from the config at every call.
        "muse_spark" => Box::new(MuseSparkVisionProcessor::new()),
        other => return Err(format!("unknown processor {other:?}: one of {PROCESSORS}").into()),
    })
}

/// SHA-256 of `bytes` as lowercase hex, the digest the recorded fixtures carry.
fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            // Writing to a String cannot fail.
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// The decoded pixels of encoded image bytes, as the gateway decodes an inline image: the
/// header gives the frame its size, the frame decodes on first use with EXIF orientation
/// applied and no size cap.
fn decode(bytes: Vec<u8>) -> Result<DynamicImage, String> {
    let (width, height) = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .into_dimensions()
        .map_err(|e| e.to_string())?;
    let frame = ImageFrame::encoded(
        bytes::Bytes::from(bytes),
        ImageDetail::Auto,
        ImageSource::InlineBytes,
        String::new(),
        ImageSize::new(width, height),
        None,
    );
    frame.image().cloned().map_err(|e| e.to_string())
}

/// Everything the pipeline gives for one image.
struct Artifacts {
    width: u32,
    height: u32,
    alpha: bool,
    grid: Vec<i64>,
    shape: Vec<usize>,
    rgb: Vec<u8>,
    u8: Option<Vec<u8>>,
    bf16: Vec<u8>,
    f32: Vec<u8>,
}

impl Artifacts {
    /// The per-image line without the digests: what `raw` mode's `meta.json` carries.
    fn describe(&self, name: &str, path: &str) -> Value {
        json!({
            "name": name,
            "path": path,
            "w": self.width,
            "h": self.height,
            "alpha": self.alpha,
            "grid": self.grid,
            "shape": self.shape,
        })
    }

    /// The `hash` line: the description and the four digests.
    fn line(&self, name: &str, path: &str) -> Value {
        let mut line = self.describe(name, path);
        line["rgb_sha256"] = json!(sha256(&self.rgb));
        line["u8_sha256"] = json!(self.u8.as_deref().map(sha256));
        line["bf16_sha256"] = json!(sha256(&self.bf16));
        line["f32_sha256"] = json!(sha256(&self.f32));
        line
    }

    /// The artifacts as files under `dir`.
    fn write(&self, dir: &Path, name: &str) -> Result<(), String> {
        let mut files = vec![
            (format!("{name}.rgb.raw"), &self.rgb),
            (format!("{name}.bf16.raw"), &self.bf16),
            (format!("{name}.f32.raw"), &self.f32),
        ];
        if let Some(u8) = &self.u8 {
            files.push((format!("{name}.u8.raw"), u8));
        }
        for (file, data) in files {
            let path = dir.join(file);
            fs::write(&path, data).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(())
    }
}

/// The image grid the processor reports, whichever integer tensor it uses for it.
fn grid(value: Option<&ModelSpecificValue>) -> Vec<i64> {
    match value {
        Some(ModelSpecificValue::IntTensor { data, .. }) => data.clone(),
        Some(ModelSpecificValue::UintTensor { data, .. }) => {
            data.iter().map(|&v| i64::from(v)).collect()
        }
        _ => Vec::new(),
    }
}

/// Decode and preprocess one image.
fn process(
    processor: &dyn VisionPreProcessor,
    config: &PreProcessorConfig,
    bytes: Vec<u8>,
) -> Result<Artifacts, String> {
    let image = decode(bytes)?;
    let alpha = image.color().has_alpha();
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let inputs = processor
        .preprocess(std::slice::from_ref(&image), config)
        .map_err(|e| e.to_string())?;
    let u8 = match &inputs.encoder_input {
        EncoderInput::U8 { patches, .. } => {
            Some(patches.as_standard_layout().iter().copied().collect())
        }
        // A wire input holds the same patch layout, already in the destination's
        // dtype: the bytes are the pixel values themselves only for a u8 wire.
        EncoderInput::Wire {
            bytes,
            dtype: EncoderDtype::Uint8,
            ..
        } => Some(bytes.clone()),
        EncoderInput::Wire { .. } | EncoderInput::F32(_) => None,
    };
    Ok(Artifacts {
        width,
        height,
        alpha,
        grid: grid(inputs.model_specific.get("image_grid_thw")),
        shape: inputs.encoder_input.shape().to_vec(),
        rgb: rgb.into_raw(),
        u8,
        bf16: inputs.encoder_input.write_as(EncoderDtype::BFloat16),
        f32: inputs.encoder_input.write_as(EncoderDtype::Float32),
    })
}

/// One image through the pipeline, with a panic in it reported as that image's error.
fn one(
    processor: &dyn VisionPreProcessor,
    config: &PreProcessorConfig,
    path: &Path,
) -> Result<Artifacts, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    match panic::catch_unwind(AssertUnwindSafe(|| process(processor, config, bytes))) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            Err(format!("PANIC: {message}"))
        }
    }
}

/// The listed paths with their names.
fn listed(list: &Path) -> Result<Vec<(String, PathBuf)>, Fault> {
    let text = fs::read_to_string(list).map_err(|e| format!("{}: {e}", list.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let path = PathBuf::from(line);
            let name = path.file_name().map_or_else(
                || line.to_string(),
                |name| name.to_string_lossy().into_owned(),
            );
            (name, path)
        })
        .collect())
}

/// The stem each listed file's artifacts are written under in `raw` mode: its name, or, where any
/// two listed files share a name, its position in the list and its name, for every file. The
/// positions are distinct, so the stems are too, whatever the names (a file named like a prefixed
/// stem included).
fn artifact_stems(names: &[String]) -> Vec<String> {
    let distinct: HashSet<&str> = names.iter().map(String::as_str).collect();
    if distinct.len() == names.len() {
        return names.to_vec();
    }
    names
        .iter()
        .enumerate()
        .map(|(index, name)| format!("{}-{name}", index + 1))
        .collect()
}

/// Where the lines go: a file in `hash` mode, the directory's `meta.json` in `raw` mode.
enum Sink {
    File(BufWriter<File>),
    Meta(Vec<Value>),
}

#[expect(
    clippy::print_stderr,
    reason = "a setup fault is the example's diagnostic output"
)]
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(fault) => {
            eprintln!("bellwether_dump: {fault}");
            ExitCode::FAILURE
        }
    }
}

#[expect(
    clippy::print_stderr,
    reason = "the run's summary line is the example's diagnostic output"
)]
fn run() -> Result<(), Fault> {
    let Some(options) = Options::parse(env::args().skip(1))? else {
        return Ok(());
    };
    let config_text = fs::read_to_string(&options.config)
        .map_err(|e| format!("{}: {e}", options.config.display()))?;
    let config = PreProcessorConfig::from_json(&config_text)
        .map_err(|e| format!("{}: {e}", options.config.display()))?;
    let processor = build_processor(&options.processor, &config)?;
    let images = listed(&options.list)?;
    let mut sink = match options.mode {
        Mode::Hash => {
            let file = File::create(&options.out)
                .map_err(|e| format!("{}: {e}", options.out.display()))?;
            Sink::File(BufWriter::new(file))
        }
        Mode::Raw => {
            fs::create_dir_all(&options.out)
                .map_err(|e| format!("{}: {e}", options.out.display()))?;
            Sink::Meta(Vec::with_capacity(images.len()))
        }
    };
    let names: Vec<String> = images.iter().map(|(name, _)| name.clone()).collect();
    let stems = artifact_stems(&names);
    let (mut ok, mut failed) = (0usize, 0usize);
    for ((name, path), stem) in images.iter().zip(&stems) {
        let path_text = path.to_string_lossy();
        let result =
            one(processor.as_ref(), &config, path).and_then(|artifacts| match options.mode {
                Mode::Hash => Ok(artifacts.line(name, &path_text)),
                Mode::Raw => artifacts.write(&options.out, stem).map(|()| {
                    let mut described = artifacts.describe(name, &path_text);
                    described["artifact"] = json!(stem);
                    described
                }),
            });
        let line = match result {
            Ok(line) => {
                ok += 1;
                line
            }
            Err(error) => {
                failed += 1;
                json!({"name": name, "path": path_text, "error": error})
            }
        };
        match &mut sink {
            Sink::File(file) => writeln!(file, "{line}")?,
            Sink::Meta(lines) => lines.push(line),
        }
    }
    match sink {
        Sink::File(mut file) => file.flush()?,
        Sink::Meta(lines) => {
            let meta = json!({"processor": options.processor, "images": lines});
            fs::write(options.out.join("meta.json"), serde_json::to_string(&meta)?)?;
        }
    }
    eprintln!(
        "DUMP_DONE mode={} processor={} ok={ok} err={failed}",
        options.mode.name(),
        options.processor
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| (*name).to_owned()).collect()
    }

    fn artifacts(byte: u8) -> Artifacts {
        Artifacts {
            width: 1,
            height: 1,
            alpha: false,
            grid: vec![1, 1, 1],
            shape: vec![1, 3],
            rgb: vec![byte; 3],
            u8: None,
            bf16: vec![byte; 6],
            f32: vec![byte; 12],
        }
    }

    #[test]
    fn distinct_names_are_the_stems() {
        assert_eq!(
            artifact_stems(&names(&["x.png", "y.png"])),
            ["x.png", "y.png"]
        );
    }

    #[test]
    fn a_shared_name_gives_every_file_its_position() {
        assert_eq!(
            artifact_stems(&names(&["x.png", "y.png", "x.png"])),
            ["1-x.png", "2-y.png", "3-x.png"]
        );
        // A file named like a prefixed stem cannot take another file's stem: positions are distinct.
        let stems = artifact_stems(&names(&["x.png", "1-x.png", "x.png"]));
        assert_eq!(stems, ["1-x.png", "2-1-x.png", "3-x.png"]);
        assert_eq!(stems.iter().collect::<HashSet<_>>().len(), stems.len());
    }

    #[test]
    fn two_images_of_one_name_in_two_directories_write_two_sets_of_files() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let list = dir.path().join("list.txt");
        fs::write(&list, "a/same.png\nb/same.png\n").expect("the list");
        let listed = listed(&list).expect("two listed files");
        let names: Vec<String> = listed.iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(names, ["same.png", "same.png"]);
        let stems = artifact_stems(&names);
        assert_eq!(stems, ["1-same.png", "2-same.png"]);
        let out = dir.path().join("raw");
        fs::create_dir_all(&out).expect("the output directory");
        artifacts(1)
            .write(&out, &stems[0])
            .expect("the first image's files");
        artifacts(2)
            .write(&out, &stems[1])
            .expect("the second image's files");
        assert_eq!(
            fs::read(out.join("1-same.png.rgb.raw")).expect("the first rgb"),
            vec![1; 3]
        );
        assert_eq!(
            fs::read(out.join("2-same.png.rgb.raw")).expect("the second rgb"),
            vec![2; 3]
        );
        assert_eq!(fs::read_dir(&out).expect("the output directory").count(), 6);
    }
}
