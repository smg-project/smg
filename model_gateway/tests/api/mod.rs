//! API endpoint integration tests

mod api_endpoints_test;
mod content_encoding_test;
mod messages_api_test;
mod messages_streaming_test;
mod messages_test;
mod messages_threads_test;
mod parser_endpoints_test;
mod path_refusals_conversations_test;
mod query_path_refusals_test;
mod request_formats_test;
#[cfg(feature = "provider-openai")]
mod responses_api_test;
mod stored_responses_query_test;
mod streaming_tests;
