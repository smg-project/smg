//! Request context types for OpenAI router pipeline.

use std::sync::Arc;

use axum::http::HeaderMap;
use openai_protocol::{chat::ChatCompletionRequest, responses::ResponsesRequest};
use serde_json::Value;
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpOrchestrator, McpToolSession};

use super::provider::Provider;
use crate::{openai_bridge, tenant::TenantRequestMeta, worker::ExternalWorker};

pub struct RequestContext {
    pub input: RequestInput,
    pub components: ComponentRefs,
    pub state: ProcessingState,
    pub storage_request_context: Option<StorageRequestContext>,
    /// Explicit tenant identity resolved at the HTTP boundary.
    pub tenant_request_meta: Option<TenantRequestMeta>,
}

pub struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    pub model_id: Option<String>,
}

pub enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Responses(Arc<ResponsesRequest>),
}

#[derive(Clone)]
pub struct SharedComponents {
    pub client: reqwest::Client,
}

pub struct ResponsesComponents {
    pub shared: Arc<SharedComponents>,
    pub mcp_orchestrator: Arc<McpOrchestrator>,
    pub mcp_format_registry: openai_bridge::FormatRegistry,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
}

pub enum ComponentRefs {
    Shared(Arc<SharedComponents>),
    Responses(Arc<ResponsesComponents>),
}

impl ComponentRefs {
    pub fn client(&self) -> &reqwest::Client {
        match self {
            ComponentRefs::Shared(s) => &s.client,
            ComponentRefs::Responses(r) => &r.shared.client,
        }
    }

    /// The MCP orchestrator, present only on the Responses path.
    pub fn mcp_orchestrator(&self) -> Option<&Arc<McpOrchestrator>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.mcp_orchestrator),
        }
    }

    pub fn mcp_format_registry(&self) -> Option<&openai_bridge::FormatRegistry> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.mcp_format_registry),
        }
    }

    pub fn response_storage(&self) -> Option<&Arc<dyn ResponseStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.response_storage),
        }
    }

    pub fn conversation_storage(&self) -> Option<&Arc<dyn ConversationStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.conversation_storage),
        }
    }

    pub fn conversation_item_storage(&self) -> Option<&Arc<dyn ConversationItemStorage>> {
        match self {
            ComponentRefs::Shared(_) => None,
            ComponentRefs::Responses(r) => Some(&r.conversation_item_storage),
        }
    }
}

#[derive(Default)]
pub struct ProcessingState {
    pub worker: Option<WorkerSelection>,
    pub payload: Option<PayloadState>,
    pub responses_payload: Option<ResponsesPayloadState>,
}

pub struct WorkerSelection {
    pub worker: Arc<dyn ExternalWorker>,
    pub provider: Arc<dyn Provider>,
}

pub struct PayloadState {
    pub json: Value,
    pub url: String,
}

#[derive(Default)]
pub struct ResponsesPayloadState {
    pub previous_response_id: Option<String>,
    pub existing_mcp_list_tools_labels: Vec<String>,
}

impl RequestContext {
    /// Build request context for Responses API calls.
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: ComponentRefs,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Responses(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
            storage_request_context: None,
            tenant_request_meta: None,
        }
    }

    /// Build request context for Chat Completions calls.
    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: ComponentRefs,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Chat(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
            storage_request_context: None,
            tenant_request_meta: None,
        }
    }
}

impl RequestContext {
    pub fn responses_request(&self) -> Option<&ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Some(req.as_ref()),
            RequestType::Chat(_) => None,
        }
    }

    pub fn responses_request_arc(&self) -> Option<Arc<ResponsesRequest>> {
        match &self.input.request_type {
            RequestType::Responses(req) => Some(Arc::clone(req)),
            RequestType::Chat(_) => None,
        }
    }

    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
        }
    }

    pub fn headers(&self) -> Option<&HeaderMap> {
        self.input.headers.as_ref()
    }

    pub fn model_id(&self) -> Option<&str> {
        self.input.model_id.as_deref()
    }

    pub fn worker(&self) -> Option<&Arc<dyn ExternalWorker>> {
        self.state.worker.as_ref().map(|w| &w.worker)
    }

    pub fn provider(&self) -> Option<&dyn Provider> {
        self.state.worker.as_ref().map(|w| w.provider.as_ref())
    }

    pub fn payload(&self) -> Option<&PayloadState> {
        self.state.payload.as_ref()
    }

    pub fn take_payload(&mut self) -> Option<PayloadState> {
        self.state.payload.take()
    }

    pub fn take_responses_payload(&mut self) -> Option<ResponsesPayloadState> {
        self.state.responses_payload.take()
    }
}

pub struct StorageHandles {
    pub response: Arc<dyn ResponseStorage>,
    pub conversation: Arc<dyn ConversationStorage>,
    pub conversation_item: Arc<dyn ConversationItemStorage>,
    pub request_context: Option<StorageRequestContext>,
}

pub struct OwnedStreamingContext {
    pub url: String,
    pub payload: Value,
    pub original_body: ResponsesRequest,
    pub previous_response_id: Option<String>,
    pub existing_mcp_list_tools_labels: Vec<String>,
    pub storage: StorageHandles,
}

impl RequestContext {
    pub fn into_streaming_context(mut self) -> Result<OwnedStreamingContext, &'static str> {
        let payload_state = self.take_payload().ok_or("Payload not prepared")?;
        let responses_payload_state = self.take_responses_payload().unwrap_or_default();
        let original_body = self
            .responses_request()
            .ok_or("Expected responses request")?
            .clone();
        let response = self
            .components
            .response_storage()
            .ok_or("Response storage required")?
            .clone();
        let conversation = self
            .components
            .conversation_storage()
            .ok_or("Conversation storage required")?
            .clone();
        let conversation_item = self
            .components
            .conversation_item_storage()
            .ok_or("Conversation item storage required")?
            .clone();

        Ok(OwnedStreamingContext {
            url: payload_state.url,
            payload: payload_state.json,
            original_body,
            previous_response_id: responses_payload_state.previous_response_id,
            existing_mcp_list_tools_labels: responses_payload_state.existing_mcp_list_tools_labels,
            storage: StorageHandles {
                response,
                conversation,
                conversation_item,
                request_context: self.storage_request_context,
            },
        })
    }
}

pub struct StreamingEventContext<'a> {
    pub original_request: &'a ResponsesRequest,
    pub previous_response_id: Option<&'a str>,
    pub session: Option<&'a McpToolSession<'a>>,
    pub mcp_format_registry: Option<&'a openai_bridge::FormatRegistry>,
}

pub type StreamingRequest = OwnedStreamingContext;
