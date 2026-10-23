# Changelog

All notable changes to `openai-protocol` are recorded here, newest first.

## Unreleased

### Removed

- Breaking: the fifty-two builder constructors nothing in this repository uses are removed, with
  the imports only they used; the change before this one deprecated them. They are: the
  `ClientEvent` constructors `conversation_item_delete`, `conversation_item_retrieve`,
  `conversation_item_truncate`, `input_audio_buffer_append`, `input_audio_buffer_clear`,
  `input_audio_buffer_commit`, `output_audio_buffer_clear`, `response_cancel`; the `ServerEvent`
  constructors `session_updated`, `conversation_created`, `conversation_item_created`,
  `conversation_item_added`, `conversation_item_done`, `conversation_item_deleted`,
  `conversation_item_retrieved`, `conversation_item_truncated`,
  `input_audio_transcription_completed`, `input_audio_transcription_completed_with_logprobs`,
  `input_audio_transcription_delta`, `input_audio_transcription_failed`,
  `input_audio_transcription_segment`, `input_audio_buffer_cleared`, `input_audio_buffer_committed`,
  `input_audio_buffer_speech_started`, `input_audio_buffer_speech_stopped`,
  `input_audio_buffer_timeout_triggered`, `dtmf_event_received`, `mcp_list_tools_in_progress`,
  `mcp_list_tools_completed`, `mcp_list_tools_failed`, `mcp_call_in_progress`, `mcp_call_completed`,
  `mcp_call_failed`, `rate_limits_updated`, `response_created`, `response_done`;
  `ResponseEventBuilder::output_audio_buffer_stopped`,
  `ResponseEventBuilder::output_audio_buffer_cleared`; `ItemEventBuilder::output_item_added`,
  `ItemEventBuilder::function_call_arguments_delta`,
  `ItemEventBuilder::function_call_arguments_done`, `ItemEventBuilder::mcp_call_arguments_delta`,
  `ItemEventBuilder::mcp_call_arguments_done`; `ContentEventBuilder::content_part_added`,
  `ContentEventBuilder::content_part_done`, `ContentEventBuilder::output_text_done`,
  `ContentEventBuilder::output_audio_delta`, `ContentEventBuilder::output_audio_done`,
  `ContentEventBuilder::output_audio_transcript_delta`,
  `ContentEventBuilder::output_audio_transcript_done`; and
  `ChatCompletionStreamResponseBuilder::add_choice_tool_args`,
  `ChatCompletionStreamResponseBuilder::add_choice_reasoning`.
