`chat_template.jinja` is the unmodified Z.ai GLM-5.3 template from
https://huggingface.co/zai-org/GLM-5.3/blob/aca966e/chat_template.jinja
(GLM-5.3 license). It is kept locally to verify the media reminder behavior
without network access.

`chat_template_vllm.jinja`, `chat_template_flash.jinja` and the 73 input cases
in `render_fixtures.json` come from [vLLM PR #62424](https://github.com/vllm-project/vllm/pull/62424),
commit `4d2f212130553e3c9c1f71684487ec06ee2b9fb4` (Apache-2.0 for the native
implementation and tests; model templates retain the model license).
`chat_template_flash_original.jinja` comes from the original Flash tokenizer
assets in `glm-moe-converters/legacy-assets/2608-vision` (model license).

The four templates identify supported native-renderer revisions. Tests compare
native output against SMG's MiniJinja renderer, including loader-transformed
templates. The corpus omits upstream expected outputs, which differ from SMG's
role normalization and continuation behavior.

Run the focused tests with `cargo test -p llm-tokenizer --lib glm`.
