use async_trait::async_trait;
use axum::response::Response;

use super::scoring::prepare_decisions;
use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext, RequestType},
        regular::symphony,
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
        // The table that reads the model replays each rendered prompt: one that
        // leaves the model inside its thought at the answer position is refused.
        let model = &ctx.input.model_id;
        let names = ctx.components.parser_resolver.names_for(None, model);
        let format = symphony::reads(&names, model, &ctx.components.templates)
            .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
        let (items, scoring) =
            prepare_decisions(&request, tokenizer, |prompt| format.reasoning_open(prompt))
                .await
                .map_err(|message| error::bad_request("unsupported_decisions_request", message))?;
        ctx.state.preparation = Some(PreparationOutput::Decisions { items, scoring });
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DecisionsPreparation"
    }
}
