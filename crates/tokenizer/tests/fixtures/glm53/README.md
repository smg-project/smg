`chat_template.jinja` and `chat_template_flash.jinja` are the unmodified Z.ai
templates from the `main` branches (at the commits below) of
https://huggingface.co/zai-org/GLM-5.3/blob/ba32f8c/chat_template.jinja and
https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050b/chat_template.jinja
(GLM-5.3 license). They are kept locally to verify the native renderer and
the media reminder behavior without network access.

The 73 input cases in `render_fixtures.json` come from [vLLM PR #62424](https://github.com/vllm-project/vllm/pull/62424),
commit `4d2f212130553e3c9c1f71684487ec06ee2b9fb4` (Apache-2.0).

The two templates identify the supported native-renderer revision; any other
template keeps MiniJinja. Tests compare native output against SMG's MiniJinja
renderer, including loader-transformed templates. The corpus omits upstream
expected outputs, which differ from SMG's role normalization and continuation
behavior.

Run the focused tests with `cargo test -p llm-tokenizer --lib glm`.
