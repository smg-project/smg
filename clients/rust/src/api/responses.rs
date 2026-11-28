use openai_protocol::responses::{ResponsesRequest, ResponsesResponse};

use crate::{transport::Transport, SmgError};

/// Responses API (`/v1/responses`).
pub struct Responses {
    transport: Transport,
}

impl Responses {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Create a new response.
    pub async fn create(&self, request: &ResponsesRequest) -> Result<ResponsesResponse, SmgError> {
        let resp = self.transport.post("/v1/responses", request).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Retrieve an existing response by ID.
    pub async fn get(&self, response_id: &str) -> Result<ResponsesResponse, SmgError> {
        let resp = self
            .transport
            .get(&format!("/v1/responses/{response_id}"))
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Delete a response by ID.
    ///
    /// Note: this endpoint is not yet implemented on the server (returns 501).
    pub async fn delete(&self, response_id: &str) -> Result<(), SmgError> {
        self.transport
            .delete(&format!("/v1/responses/{response_id}"))
            .await?;
        Ok(())
    }

    /// Cancel an in-progress response.
    pub async fn cancel(&self, response_id: &str) -> Result<ResponsesResponse, SmgError> {
        let resp = self
            .transport
            .post(
                &format!("/v1/responses/{response_id}/cancel"),
                &serde_json::json!({}),
            )
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// List input items for a response: the first page with the route's
    /// defaults (a page of 20, the newest item first).
    pub async fn list_input_items(&self, response_id: &str) -> Result<serde_json::Value, SmgError> {
        self.list_input_items_page(response_id, &InputItemsQuery::default())
            .await
    }

    /// List one page of a response's input items: the answer carries `data`,
    /// `first_id`, `last_id` and `has_more`; the next page is the same call
    /// with `after` set to this page's `last_id`.
    pub async fn list_input_items_page(
        &self,
        response_id: &str,
        query: &InputItemsQuery,
    ) -> Result<serde_json::Value, SmgError> {
        let resp = self
            .transport
            .get_with_query(
                &format!("/v1/responses/{response_id}/input_items"),
                &query.pairs(),
            )
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}

/// The order of an input-items page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputItemsOrder {
    /// The order the request gave the items.
    Asc,
    /// The newest item first (the route's default).
    Desc,
}

impl InputItemsOrder {
    fn as_str(self) -> &'static str {
        match self {
            Self::Asc => "asc",
            Self::Desc => "desc",
        }
    }
}

/// The query of [`Responses::list_input_items_page`]; a field left `None` or
/// empty sends nothing and the route applies its default (a page of 20, the
/// newest first, from the first item, no extra field).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InputItemsQuery {
    /// The page size, 1 to 100.
    pub limit: Option<u32>,
    /// The page's order.
    pub order: Option<InputItemsOrder>,
    /// The id of the item the page follows (the previous page's `last_id`).
    pub after: Option<String>,
    /// The extra fields to include, as the route names them; each is sent as
    /// its own `include` parameter.
    pub include: Vec<String>,
}

impl InputItemsQuery {
    /// The query pairs, in the order the public API documents them.
    fn pairs(&self) -> Vec<(&'static str, String)> {
        let mut pairs = Vec::with_capacity(3 + self.include.len());
        if let Some(limit) = self.limit {
            pairs.push(("limit", limit.to_string()));
        }
        if let Some(order) = self.order {
            pairs.push(("order", order.as_str().to_string()));
        }
        if let Some(after) = &self.after {
            pairs.push(("after", after.clone()));
        }
        pairs.extend(self.include.iter().map(|field| ("include", field.clone())));
        pairs
    }
}

#[cfg(test)]
#[path = "../../tests/unit/api/responses/tests.rs"]
mod tests;
