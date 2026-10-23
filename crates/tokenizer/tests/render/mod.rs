//! Chat-template rendering integration tests: the template formats, the renderers a checkpoint
//! selects and their parity with the reference renders.

mod chat_template_continuation;
mod chat_template_format_detection;
mod chat_template_integration;
mod chat_template_loading;
mod chat_template_openai_string_content;
mod deepseek_renderer_detection;
mod deepseek_v31_tools_block;
mod deepseek_v41_renderer;
mod deepseek_v4_renderer;
mod kimi_k25_renderer_detection;
mod kimi_k3_renderer;
mod minimax_root_role_renderer;
