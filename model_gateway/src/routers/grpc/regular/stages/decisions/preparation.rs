use async_trait::async_trait;
use axum::response::Response;

use super::scoring::prepare_decisions;
use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext, RequestType},
        utils,
    },
};

pub(crate) struct DecisionsPreparationStage;

#[async_trait]
impl PipelineStage for DecisionsPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let RequestType::Decisions(request) = &ctx.input.request_type else {
            return Err(error::internal_error(
                "invalid_request_type",
                "Expected Decisions request",
            ));
        };
        let request = request.clone();
        let tokenizer = utils::resolve_tokenizer(ctx, "DecisionsPreparationStage::execute")
            .map_err(|error| *error)?;
        let configured = ctx
            .components
            .parser_resolver
            .reasoning_parser(&ctx.input.model_id);
        let answers_open_reasoning = utils::create_reasoning_parser(
            &ctx.components.reasoning_parser_factory,
            configured.as_deref(),
            &ctx.input.model_id,
        )
        .is_some_and(|parser| parser.is_in_reasoning());
        let (items, scoring) = prepare_decisions(&request, tokenizer, answers_open_reasoning)
            .await
            .map_err(|message| error::bad_request("unsupported_decisions_request", message))?;
        ctx.state.preparation = Some(PreparationOutput::Decisions { items, scoring });
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DecisionsPreparation"
    }
}
