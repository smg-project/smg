//! Request capture: `--capture` records every gRPC `Generate` request the
//! worker receives, one JSON object per line, so a test can check what the
//! gateway put on the wire. (The trace replayer is this crate's `replay`
//! binary, `src/bin/replay.rs`.)

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{self, Write},
    path::Path,
    sync::{Mutex, PoisonError},
};

use serde_json::{json, Map, Number, Value};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use ts::sampling_params::Constraint;

/// An append-mode capture file holding one JSON object per `Generate` request.
pub struct Capture {
    file: Mutex<File>,
}

impl Capture {
    /// Open `path` for appending, creating it if it does not exist. A capture
    /// holds prompts, so a file this creates is readable by its owner only
    /// (mode 0o600 on Unix). A file that already exists keeps its mode: it
    /// was made by someone who chose who may read it.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options.open(path)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    /// Append `req` as one line. The line is in the file when this returns:
    /// `File` keeps no buffer of its own, so a reader that opens the file
    /// afterwards sees the line, and killing the process cannot lose it.
    pub fn record(&self, req: &ts::GenerateRequest) -> io::Result<()> {
        let mut line = serde_json::to_vec(&capture_json(req))?;
        line.push(b'\n');
        // One `write_all` under the lock keeps this worker's lines whole and
        // in order. Append mode puts each write at the current end of the
        // file, so workers that share the file add to it, never overwrite it.
        let mut file = self.file.lock().unwrap_or_else(PoisonError::into_inner);
        file.write_all(&line)?;
        file.flush()
    }
}

/// The capture line for `req`, written by hand because the prost types do not
/// derive `Serialize`. Keys are the proto field names. An optional the
/// client left unset is `null`; an absent `tokenized` or `sampling_params`
/// message reads as its defaults, as it does for an engine. Fields that
/// carry payloads (`custom_params`, `mm_inputs`, the bootstrap infos) and
/// `data_parallel_rank` are recorded only as present or absent.
fn capture_json(req: &ts::GenerateRequest) -> Value {
    let (input_ids, original_text) = match &req.tokenized {
        Some(tokenized) => (
            tokenized.input_ids.as_slice(),
            tokenized.original_text.as_str(),
        ),
        None => (&[][..], ""),
    };
    let unset = ts::SamplingParams::default();
    let sampling = req.sampling_params.as_ref().unwrap_or(&unset);
    // Decoding puts `logit_bias` in a `HashMap`, whose order changes from one
    // decode to the next; sorted keys give the same request the same bytes.
    let logit_bias: Map<String, Value> = sampling
        .logit_bias
        .iter()
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|(token, bias)| (token.clone(), f32_json(*bias)))
        .collect();
    json!({
        "request_id": req.request_id,
        "input_ids": input_ids,
        "original_text": original_text,
        "stream": req.stream,
        "temperature": sampling.temperature.map(f32_json),
        "top_p": sampling.top_p.map(f32_json),
        "top_k": sampling.top_k,
        "min_p": sampling.min_p.map(f32_json),
        "repetition_penalty": sampling.repetition_penalty.map(f32_json),
        "frequency_penalty": sampling.frequency_penalty.map(f32_json),
        "presence_penalty": sampling.presence_penalty.map(f32_json),
        "max_new_tokens": sampling.max_new_tokens,
        "min_new_tokens": sampling.min_new_tokens,
        "stop": sampling.stop,
        "stop_token_ids": sampling.stop_token_ids,
        "ignore_eos": sampling.ignore_eos,
        "no_stop_trim": sampling.no_stop_trim,
        "skip_special_tokens": sampling.skip_special_tokens,
        "spaces_between_special_tokens": sampling.spaces_between_special_tokens,
        "n": sampling.n,
        "sampling_seed": sampling.sampling_seed,
        "logit_bias": logit_bias,
        "constraint": constraint_json(sampling.constraint.as_ref()),
        "return_logprob": req.return_logprob,
        "logprob_start_len": req.logprob_start_len,
        "top_logprobs_num": req.top_logprobs_num,
        "token_ids_logprob": req.token_ids_logprob,
        "has_custom_params": sampling.custom_params.is_some(),
        "has_mm_inputs": req.mm_inputs.is_some(),
        "has_encode_bootstrap_info": req.encode_bootstrap_info.is_some(),
        "has_kv_bootstrap_info": req.kv_bootstrap_info.is_some(),
        "has_data_parallel_rank": req.data_parallel_rank.is_some(),
    })
}

/// An `f32` as the shortest decimal that reads back as the same `f32`, so a
/// request's `0.7` is captured as `0.7`, not as the `0.699999988079071` that
/// widening it to `f64` prints. JSON has no NaN or infinity, and `null`
/// means unset, so a non-finite value is written as its name, e.g. `"NaN"`.
fn f32_json(value: f32) -> Value {
    let shortest = value.to_string();
    match shortest.parse::<f64>().ok().and_then(Number::from_f64) {
        Some(number) => Value::Number(number),
        None => Value::String(shortest),
    }
}

/// `{"kind", "value"}` for the constraint oneof, with the value as sent.
fn constraint_json(constraint: Option<&Constraint>) -> Value {
    let (kind, value) = match constraint {
        None => return Value::Null,
        Some(Constraint::Regex(value)) => ("regex", value),
        Some(Constraint::JsonSchema(value)) => ("json_schema", value),
        Some(Constraint::EbnfGrammar(value)) => ("ebnf_grammar", value),
        Some(Constraint::StructuralTag(value)) => ("structural_tag", value),
    };
    json!({ "kind": kind, "value": value })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JSON has no NaN or infinity, and `null` means unset, so a set
    /// non-finite value must come back as something other than `null`.
    #[test]
    fn non_finite_floats_are_not_captured_as_unset() {
        assert_eq!(f32_json(f32::NAN), json!("NaN"));
        assert_eq!(f32_json(f32::INFINITY), json!("inf"));
        assert_eq!(f32_json(f32::NEG_INFINITY), json!("-inf"));
    }

    /// A capture holds prompts, so a file the mock creates is readable by
    /// its owner only. A file that already exists keeps the mode it has.
    #[cfg(unix)]
    #[test]
    fn a_new_capture_file_is_owner_only() {
        use std::{fs, os::unix::fs::PermissionsExt};

        let dir = tempfile::tempdir().expect("temp dir");
        let mode = |path: &Path| {
            let mode = fs::metadata(path)
                .expect("capture file metadata")
                .permissions()
                .mode();
            format!("{:o}", mode & 0o777)
        };

        let created = dir.path().join("created.jsonl");
        Capture::open(&created).expect("create a capture file");
        assert_eq!(mode(&created), "600", "a new file is owner-only");

        let existing = dir.path().join("existing.jsonl");
        fs::write(&existing, b"").expect("create a file");
        fs::set_permissions(&existing, fs::Permissions::from_mode(0o644)).expect("chmod 644");
        Capture::open(&existing).expect("open an existing capture file");
        assert_eq!(mode(&existing), "644", "an existing file keeps its mode");
    }
}
