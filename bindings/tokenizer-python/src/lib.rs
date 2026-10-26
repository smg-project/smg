use std::{collections::HashMap, sync::Arc};

use llm_tokenizer::{
    chat_template::ChatTemplateParams, create_tokenizer_with_chat_template, HuggingFaceTokenizer,
    PromptEncoding, TokenizerTrait,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::Value;

type RenderOutput = (String, Vec<u32>, Option<Vec<(usize, usize)>>, bool);

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(frozen)]
struct Tokenizer {
    inner: Arc<dyn TokenizerTrait>,
}

#[pymethods]
impl Tokenizer {
    #[new]
    #[pyo3(signature = (path, chat_template_path=None))]
    fn new(py: Python<'_>, path: &str, chat_template_path: Option<&str>) -> PyResult<Self> {
        let inner = py
            .detach(|| create_tokenizer_with_chat_template(path, chat_template_path))
            .map_err(value_error)?;
        Ok(Self { inner })
    }

    fn encode(&self, py: Python<'_>, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        py.detach(|| self.inner.encode(text, add_special_tokens))
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(value_error)
    }

    #[getter]
    fn encode_path(&self) -> String {
        self.inner
            .as_any()
            .downcast_ref::<HuggingFaceTokenizer>()
            .map_or_else(
                || "tiktoken".to_owned(),
                |hf| hf.encode_path().as_str().to_owned(),
            )
    }

    #[getter]
    fn raw_tool_call_arguments(&self) -> bool {
        self.inner.renderer_capabilities().raw_tool_call_arguments
    }

    fn render_json(&self, py: Python<'_>, request: &str) -> PyResult<RenderOutput> {
        let request: Value = serde_json::from_str(request).map_err(value_error)?;
        let messages = request["messages"]
            .as_array()
            .ok_or_else(|| value_error("messages must be an array"))?;
        let tools = if request["tools"].is_null() {
            None
        } else {
            Some(
                request["tools"]
                    .as_array()
                    .ok_or_else(|| value_error("tools must be an array"))?,
            )
        };
        let kwargs: HashMap<String, Value> =
            serde_json::from_value(request["template_kwargs"].clone()).map_err(value_error)?;
        let now = match &request["now"] {
            Value::Null => None,
            Value::String(value) => {
                Some(chrono::DateTime::parse_from_rfc3339(value).map_err(value_error)?)
            }
            _ => return Err(value_error("now must be an RFC3339 timestamp or null")),
        };
        for key in [
            "add_generation_prompt",
            "continue_final_message",
            "messages",
            "tools",
        ] {
            if kwargs.contains_key(key) {
                return Err(value_error(format!(
                    "{key} cannot override the complete-conversation contract"
                )));
            }
        }
        py.detach(|| {
            let rendered = self
                .inner
                .apply_chat_template_with_encoding(
                    messages,
                    ChatTemplateParams {
                        add_generation_prompt: false,
                        tools: tools.map(Vec::as_slice),
                        template_kwargs: Some(&kwargs),
                        now,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(value_error)?;
            if rendered.generation_prompt || rendered.continued_final_message {
                return Err(value_error(
                    "renderer did not honor complete-conversation mode",
                ));
            }
            let (encoding, segmentwise) = match rendered.encoding {
                PromptEncoding::FromText => (
                    self.inner
                        .encode(&rendered.text, false)
                        .map_err(value_error)?,
                    false,
                ),
                PromptEncoding::Deferred(job) => (job.run().map_err(value_error)?, true),
            };
            // Accelerated encoders may return IDs only. Obtain byte offsets from
            // the same HF pipeline, accepting them only after exact ID equality.
            // Native segment encodings must not be assigned flat-text offsets.
            let offsets = if segmentwise {
                None
            } else if let Some(hf) = self.inner.as_any().downcast_ref::<HuggingFaceTokenizer>() {
                let aligned = hf
                    .tokenizers_pipeline()
                    .encode(rendered.text.as_str(), false)
                    .map_err(value_error)?;
                if aligned.get_ids() != encoding.token_ids() {
                    return Err(value_error(
                        "SMG encoder and offset pipeline disagree on token IDs",
                    ));
                }
                Some(aligned.get_offsets().to_vec())
            } else {
                None
            };
            Ok((
                rendered.text,
                encoding.token_ids().to_vec(),
                offsets,
                segmentwise,
            ))
        })
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add("TOKENIZERS_V1", cfg!(feature = "tokenizers-v1"))?;
    Ok(())
}
