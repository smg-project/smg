//! Factory for creating router instances

use std::sync::Arc;

use super::{
    anthropic::AnthropicRouter,
    gemini::GeminiRouter,
    grpc::{
        mode::{grpc_mode, Mode},
        router::GrpcRouter,
    },
    http::{pd_router::PDRouter, router::Router},
    openai::OpenAIRouter,
    RouterTrait,
};
use crate::{
    app_context::AppContext,
    config::{PolicyConfig, RoutingMode},
    policies::{DPRankLoadPolicy, MinimumTokensPolicy, PolicyFactory},
    worker::ConnectionMode,
};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(&'static str);

impl RouterId {
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        self.0
    }
}

/// Static router ID constants to avoid heap allocations in hot paths
pub mod router_ids {
    use super::RouterId;

    pub const HTTP_REGULAR: RouterId = RouterId::new("http-regular");
    pub const HTTP_PD: RouterId = RouterId::new("http-pd");
    pub const HTTP_OPENAI: RouterId = RouterId::new("http-openai");
    pub const HTTP_ANTHROPIC: RouterId = RouterId::new("http-anthropic");
    pub const HTTP_GEMINI: RouterId = RouterId::new("http-gemini");
    pub const GRPC_REGULAR: RouterId = RouterId::new("grpc-regular");
    pub const GRPC_PD: RouterId = RouterId::new("grpc-pd");
    pub const GRPC_EPD: RouterId = RouterId::new("grpc-epd");
}

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        match ctx.router_config.connection_mode {
            ConnectionMode::Grpc | ConnectionMode::Zmq => {
                // Register the per-role policies each disaggregation mode needs,
                // then build the mode-parameterized gRPC router.
                match &ctx.router_config.mode {
                    RoutingMode::Regular { .. } => {}
                    RoutingMode::PrefillDecode {
                        prefill_policy,
                        decode_policy,
                        ..
                    } => Self::set_pd_policies(
                        prefill_policy.as_ref(),
                        decode_policy.as_ref(),
                        &ctx.router_config.policy,
                        ctx,
                    ),
                    RoutingMode::EncodePrefillDecode {
                        encode_policy,
                        prefill_policy,
                        decode_policy,
                        ..
                    } => Self::set_epd_policies(
                        encode_policy.as_ref(),
                        prefill_policy.as_ref(),
                        decode_policy.as_ref(),
                        &ctx.router_config.policy,
                        ctx,
                    ),
                    RoutingMode::OpenAI { .. } => {
                        return Err("OpenAI mode requires HTTP connection_mode".to_string())
                    }
                    RoutingMode::Anthropic { .. } => {
                        return Err("Anthropic mode requires HTTP connection_mode".to_string())
                    }
                    RoutingMode::Gemini { .. } => {
                        return Err("Gemini mode requires HTTP connection_mode".to_string())
                    }
                }
                let mode = grpc_mode(&ctx.router_config).ok_or_else(|| {
                    "gRPC connection mode requires a gRPC routing mode".to_string()
                })?;
                Self::create_grpc_router(ctx, mode)
            }
            ConnectionMode::Http => match &ctx.router_config.mode {
                RoutingMode::Regular { .. } => Self::create_regular_router(ctx).await,
                RoutingMode::PrefillDecode {
                    prefill_policy,
                    decode_policy,
                    ..
                } => {
                    Self::create_pd_router(
                        prefill_policy.as_ref(),
                        decode_policy.as_ref(),
                        &ctx.router_config.policy,
                        ctx,
                    )
                    .await
                }
                RoutingMode::EncodePrefillDecode { .. } => {
                    Err("EPD mode requires gRPC connection_mode and TokenSpeed".to_string())
                }
                RoutingMode::OpenAI { .. } => Self::create_openai_router(ctx).await,
                RoutingMode::Anthropic { .. } => Self::create_anthropic_router(ctx).await,
                RoutingMode::Gemini { .. } => Self::create_gemini_router(ctx).await,
            },
        }
    }

    /// Create a regular router
    pub async fn create_regular_router(
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = Router::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a PD router with injected policy
    pub async fn create_pd_router(
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);

        let config = ctx.router_config.clone();
        if config.dp_minimum_tokens_scheduler {
            let mini_tokens_policy = MinimumTokensPolicy::new(
                ctx.worker_monitor
                    .as_ref()
                    .map(|monitor| monitor.worker_load_manager.clone()),
            );
            let dp_rank_policy: Arc<dyn DPRankLoadPolicy> = Arc::new(mini_tokens_policy);
            ctx.policy_registry.set_dp_rank_policy(dp_rank_policy);
        }
        let router = PDRouter::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a gRPC router for the given disaggregation `mode`.
    ///
    /// For PD/EPD modes the caller must register the prefill/decode (and encode)
    /// policies first via [`Self::set_pd_policies`] / [`Self::set_epd_policies`].
    pub(crate) fn create_grpc_router(
        ctx: &Arc<AppContext>,
        mode: Mode,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = GrpcRouter::new(ctx, mode)?;

        Ok(Box::new(router))
    }

    /// Register the prefill/decode policies for a gRPC PD deployment, defaulting
    /// to the main policy when a per-role policy is unset.
    fn set_pd_policies(
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) {
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);
    }

    /// Register the encode/prefill/decode policies for a gRPC EPD deployment.
    /// Encode defaults to consistent hashing over each item's content hash;
    /// prefill/decode default to the main policy when unset.
    fn set_epd_policies(
        encode_policy_config: Option<&PolicyConfig>,
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) {
        let default_encode_policy = PolicyConfig::ConsistentHashing;
        let encode_policy = PolicyFactory::create_from_config(
            encode_policy_config.unwrap_or(&default_encode_policy),
        );
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        ctx.policy_registry.set_encode_policy(encode_policy);
        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);
    }

    /// Create an OpenAI router
    ///
    /// Workers should be registered via the external worker registration workflow
    /// before using this router. The workflow discovers models from the provided
    /// endpoints and creates external workers in the registry.
    pub async fn create_openai_router(
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = OpenAIRouter::new(ctx).await?;
        Ok(Box::new(router))
    }

    /// Create an Anthropic router
    ///
    /// Handles Anthropic Messages API (/v1/messages) with support for streaming,
    /// tool use, extended thinking, and other Anthropic-specific features.
    #[expect(
        clippy::unused_async,
        reason = "async for API consistency with other create_* factory methods"
    )]
    pub async fn create_anthropic_router(
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = AnthropicRouter::new(ctx.clone())?;
        Ok(Box::new(router))
    }

    /// Create a Gemini Interactions router
    ///
    /// Handles Gemini Interactions API (/v1/interactions) with support for
    /// streaming, MCP tool interception, and native Gemini format passthrough.
    #[expect(
        clippy::unused_async,
        reason = "async for API consistency with other create_* factory methods"
    )]
    pub async fn create_gemini_router(
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = GeminiRouter::new(ctx.clone())?;
        Ok(Box::new(router))
    }

    /// Create all routers for IGW (multi-router) mode.
    ///
    /// Returns a list of (router_id, label, creation_result) tuples.
    /// Adding a new router to IGW mode only requires adding a line here.
    pub async fn create_igw_routers(
        policy: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Vec<(RouterId, &'static str, Result<Box<dyn RouterTrait>, String>)> {
        let (encode_policy, prefill_policy, decode_policy) = match &ctx.router_config.mode {
            RoutingMode::PrefillDecode {
                prefill_policy,
                decode_policy,
                ..
            } => (None, prefill_policy.as_ref(), decode_policy.as_ref()),
            RoutingMode::EncodePrefillDecode {
                encode_policy,
                prefill_policy,
                decode_policy,
                ..
            } => (
                encode_policy.as_ref(),
                prefill_policy.as_ref(),
                decode_policy.as_ref(),
            ),
            _ => (None, None, None),
        };

        vec![
            (
                router_ids::HTTP_REGULAR,
                "HTTP Regular",
                Self::create_regular_router(ctx).await,
            ),
            (
                router_ids::GRPC_REGULAR,
                "gRPC Regular",
                Self::create_grpc_router(ctx, Mode::Regular),
            ),
            (
                router_ids::HTTP_PD,
                "HTTP PD",
                Self::create_pd_router(prefill_policy, decode_policy, policy, ctx).await,
            ),
            (router_ids::GRPC_PD, "gRPC PD", {
                Self::set_pd_policies(prefill_policy, decode_policy, policy, ctx);
                Self::create_grpc_router(ctx, Mode::PrefillDecode)
            }),
            (router_ids::GRPC_EPD, "gRPC EPD", {
                Self::set_epd_policies(encode_policy, prefill_policy, decode_policy, policy, ctx);
                Self::create_grpc_router(ctx, Mode::EncodePrefillDecode)
            }),
            (
                router_ids::HTTP_OPENAI,
                "OpenAI",
                Self::create_openai_router(ctx).await,
            ),
            (
                router_ids::HTTP_ANTHROPIC,
                "Anthropic",
                Self::create_anthropic_router(ctx).await,
            ),
            (
                router_ids::HTTP_GEMINI,
                "Gemini",
                Self::create_gemini_router(ctx).await,
            ),
        ]
    }
}

#[cfg(test)]
mod grpc_router_type_tests {
    use std::sync::{Arc, OnceLock};

    use llm_tokenizer::registry::TokenizerRegistry;
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use smg_mcp::{McpConfig, McpOrchestrator};
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::*;
    use crate::{app_context::AppContext, config::RouterConfig, policies::PolicyRegistry};

    /// Build an `AppContext` for a gRPC `RoutingMode` with the components
    /// `GrpcRouter::new` needs (parser factories always; an initialized MCP
    /// orchestrator for Regular/PD, which build the responses contexts).
    async fn grpc_ctx(mode: RoutingMode) -> Arc<AppContext> {
        let config = RouterConfig::builder()
            .mode(mode)
            .grpc_connection()
            .policy(PolicyConfig::Random)
            .host("127.0.0.1")
            .port(3001)
            .max_payload_size(1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(10)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let worker_registry = Arc::new(crate::worker::WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(config.policy.clone()));
        let mcp_orchestrator = Arc::new(OnceLock::new());
        mcp_orchestrator
            .set(Arc::new(
                McpOrchestrator::new(McpConfig::default())
                    .await
                    .expect("mcp orchestrator"),
            ))
            .ok();

        Arc::new(
            AppContext::builder()
                .router_config(config)
                .client(reqwest::Client::new())
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(Some(ReasoningParserFactory::new()))
                .tool_parser_factory(Some(ToolParserFactory::new()))
                .worker_registry(worker_registry)
                .policy_registry(policy_registry)
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(mcp_orchestrator)
                .build()
                .expect("app context"),
        )
    }

    #[tokio::test]
    async fn grpc_router_type_reflects_disaggregation_mode() {
        let cases = [
            (
                RoutingMode::Regular {
                    worker_urls: vec![],
                },
                "grpc",
            ),
            (
                RoutingMode::PrefillDecode {
                    prefill_urls: vec![],
                    decode_urls: vec![],
                    prefill_policy: None,
                    decode_policy: None,
                },
                "grpc_pd",
            ),
            (
                RoutingMode::EncodePrefillDecode {
                    encode_urls: vec![],
                    prefill_urls: vec![],
                    decode_urls: vec![],
                    encode_policy: None,
                    prefill_policy: None,
                    decode_policy: None,
                },
                "grpc_epd",
            ),
        ];

        for (mode, expected) in cases {
            let ctx = grpc_ctx(mode).await;
            let router = RouterFactory::create_router(&ctx)
                .await
                .expect("router creation");
            assert_eq!(router.router_type(), expected);
        }
    }

    #[tokio::test]
    async fn igw_routers_preserve_disaggregated_role_policies() {
        for mode in [
            RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: Some(PolicyConfig::RoundRobin),
                decode_policy: Some(PolicyConfig::Passthrough),
            },
            RoutingMode::EncodePrefillDecode {
                encode_urls: vec![],
                prefill_urls: vec![],
                decode_urls: vec![],
                encode_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5,
                }),
                prefill_policy: Some(PolicyConfig::RoundRobin),
                decode_policy: Some(PolicyConfig::Passthrough),
            },
        ] {
            let expected_encode_policy = mode
                .get_encode_policy(&PolicyConfig::ConsistentHashing)
                .name();
            let ctx = grpc_ctx(mode).await;
            RouterFactory::create_igw_routers(&ctx.router_config.policy, &ctx).await;

            let policies = &ctx.policy_registry;
            assert_eq!(policies.get_encode_policy().name(), expected_encode_policy);
            assert_eq!(policies.get_prefill_policy().name(), "round_robin");
            assert_eq!(policies.get_decode_policy().name(), "passthrough");
        }
    }
}
