use async_trait::async_trait;
use axum::response::Response;
use symphony::formats::thinks_first;

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
        let answers_open_reasoning = match (&configured, thinks_first(&ctx.input.model_id)) {
            // The registry's fact for the model's group.
            (None, Some(fact)) => fact,
            // A deployment names a legacy reasoning parser, or the registry does not know the
            // model: that parser's first state answers.
            _ => utils::create_reasoning_parser(
                &ctx.components.reasoning_parser_factory,
                configured.as_deref(),
                &ctx.input.model_id,
            )
            .is_some_and(|parser| parser.is_in_reasoning()),
        };
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

#[cfg(test)]
mod tests {
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use symphony::formats::{groups, thinks_first};

    // The stage trusts this equality; a model meant to differ is written down here.
    #[test]
    fn the_registry_says_what_the_legacy_parser_said_for_every_id_it_knows() {
        let factory = ReasoningParserFactory::new();
        for (model, also, _) in groups() {
            for id in std::iter::once(model).chain(also.iter().copied()) {
                let legacy = factory
                    .registry()
                    .create_for_model(id)
                    .is_some_and(|parser| parser.is_in_reasoning());
                assert_eq!(thinks_first(id), Some(legacy), "{id}");
            }
        }
    }
}
