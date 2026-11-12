//! Shared response functionality used by both regular and harmony implementations

pub(crate) mod background;
pub(crate) mod context;
pub(crate) mod driven;
pub(crate) mod handlers;
pub(crate) mod history;
pub(crate) mod streaming;
pub(crate) mod utils;

// Re-export commonly used items
pub(crate) use background::{spawn_unstored_sweeper, BackgroundResponses, BackgroundStream};
pub(crate) use context::ResponsesContext;
#[cfg(test)]
pub(crate) use driven::DrivenBody;
pub(crate) use driven::{
    build_sse_response_driven, build_sse_response_driven_with, driven_channel, Reading, ReadingEnd,
};
pub(crate) use history::{load_history, ReplayInto};
pub(crate) use streaming::{
    await_stream_startup, body_guards, build_sse_response, build_sse_response_from_stream,
    naming_parser, signal_stream_startup, stream_startup_channel, StreamStartupSender,
};
pub(crate) use utils::{
    ensure_mcp_connection, persist_response_if_needed, previous_response_chain, stored_foreground,
    tool_call_limit, NotStored,
};

pub(crate) use crate::routers::common::mcp_utils::collect_user_function_names;

#[cfg(test)]
pub(crate) mod test_support {
    //! What the responses modules' tests share.

    use openai_protocol::responses::{ResponseStatus, ResponsesRequest, ResponsesResponse};
    use serde_json::json;

    /// The terminal object of `request` with one output message.
    pub(crate) fn completed_response(id: &str, request: &ResponsesRequest) -> ResponsesResponse {
        let message = serde_json::from_value(json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}]
        }))
        .unwrap();
        ResponsesResponse::builder(id, &request.model)
            .copy_from_request(request)
            .created_at(chrono::Utc::now().timestamp())
            .completed_at(chrono::Utc::now().timestamp())
            .status(ResponseStatus::Completed)
            .add_output(message)
            .build()
    }

    /// `impl ResponseStorage` for a test double that wraps another store: the
    /// methods written in the block as they are, and every method named after
    /// `passes` handed to the wrapped store (`Double.field`), so a double spells
    /// only the calls it takes.
    macro_rules! response_storage_over {
        ($double:ident . $inner:tt; passes $($pass:ident),* $(,)?; $($own:tt)*) => {
            response_storage_over!(@build $double, $inner, [$($pass)*], [] $($own)*);
        };
        (@build $double:ident, $inner:tt, [], [$($pass:tt)*] $($own:tt)*) => {
            #[async_trait::async_trait]
            impl smg_data_connector::ResponseStorage for $double {
                $($own)*
                $($pass)*
            }
        };
        (@build $double:ident, $inner:tt, [store_response $($rest:ident)*], [$($pass:tt)*] $($own:tt)*) => {
            response_storage_over!(@build $double, $inner, [$($rest)*], [$($pass)*
                async fn store_response(
                    &self,
                    response: smg_data_connector::StoredResponse,
                ) -> Result<smg_data_connector::ResponseId, smg_data_connector::ResponseStorageError> {
                    self.$inner.store_response(response).await
                }
            ] $($own)*);
        };
        (@build $double:ident, $inner:tt, [store_response_if_open $($rest:ident)*], [$($pass:tt)*] $($own:tt)*) => {
            response_storage_over!(@build $double, $inner, [$($rest)*], [$($pass)*
                async fn store_response_if_open(
                    &self,
                    response: smg_data_connector::StoredResponse,
                ) -> Result<smg_data_connector::ConditionalStore, smg_data_connector::ResponseStorageError> {
                    self.$inner.store_response_if_open(response).await
                }
            ] $($own)*);
        };
        (@build $double:ident, $inner:tt, [update_response_object_if_open $($rest:ident)*], [$($pass:tt)*] $($own:tt)*) => {
            response_storage_over!(@build $double, $inner, [$($rest)*], [$($pass)*
                async fn update_response_object_if_open(
                    &self,
                    id: &smg_data_connector::ResponseId,
                    raw_response: serde_json::Value,
                ) -> Result<smg_data_connector::ConditionalStore, smg_data_connector::ResponseStorageError> {
                    self.$inner
                        .update_response_object_if_open(id, raw_response)
                        .await
                }
            ] $($own)*);
        };
        (@build $double:ident, $inner:tt, [get_response $($rest:ident)*], [$($pass:tt)*] $($own:tt)*) => {
            response_storage_over!(@build $double, $inner, [$($rest)*], [$($pass)*
                async fn get_response(
                    &self,
                    response_id: &smg_data_connector::ResponseId,
                ) -> Result<Option<smg_data_connector::StoredResponse>, smg_data_connector::ResponseStorageError> {
                    self.$inner.get_response(response_id).await
                }
            ] $($own)*);
        };
        (@build $double:ident, $inner:tt, [delete_response $($rest:ident)*], [$($pass:tt)*] $($own:tt)*) => {
            response_storage_over!(@build $double, $inner, [$($rest)*], [$($pass)*
                async fn delete_response(
                    &self,
                    response_id: &smg_data_connector::ResponseId,
                ) -> Result<(), smg_data_connector::ResponseStorageError> {
                    self.$inner.delete_response(response_id).await
                }
            ] $($own)*);
        };
        (@build $double:ident, $inner:tt, [list_identifier_responses $($rest:ident)*], [$($pass:tt)*] $($own:tt)*) => {
            response_storage_over!(@build $double, $inner, [$($rest)*], [$($pass)*
                async fn list_identifier_responses(
                    &self,
                    identifier: &str,
                    tenant: Option<&str>,
                    limit: Option<usize>,
                ) -> Result<Vec<smg_data_connector::StoredResponse>, smg_data_connector::ResponseStorageError> {
                    self.$inner
                        .list_identifier_responses(identifier, tenant, limit)
                        .await
                }
            ] $($own)*);
        };
        (@build $double:ident, $inner:tt, [delete_identifier_responses $($rest:ident)*], [$($pass:tt)*] $($own:tt)*) => {
            response_storage_over!(@build $double, $inner, [$($rest)*], [$($pass)*
                async fn delete_identifier_responses(
                    &self,
                    identifier: &str,
                    tenant: Option<&str>,
                ) -> Result<usize, smg_data_connector::ResponseStorageError> {
                    self.$inner
                        .delete_identifier_responses(identifier, tenant)
                        .await
                }
            ] $($own)*);
        };
    }
    pub(crate) use response_storage_over;
}
