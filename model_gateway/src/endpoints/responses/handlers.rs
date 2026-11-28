use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use smg_data_connector::{
    ResponseId, ResponseStorage, ResponseStorageError, SortOrder, StoredResponse,
};
use tracing::{info, warn};

use crate::{endpoints::conversations::parse_item_include_query, routers::error};

/// The response stored under `id` as a caller of `tenant` may see it: `None`
/// for an id that is not stored and for one stored under another tenant, the
/// same answer, so an id tells a caller nothing about another tenant's records.
async fn visible_response(
    response_storage: &Arc<dyn ResponseStorage>,
    id: &ResponseId,
    tenant: Option<&str>,
) -> Result<Option<StoredResponse>, ResponseStorageError> {
    Ok(response_storage
        .get_response(id)
        .await?
        .filter(|stored| stored.is_visible_to(tenant)))
}

/// The code a stored-response route refuses a query value under: the one the
/// models list uses for a parameter the contract does not take, the parameter
/// named first in the message.
const INVALID_QUERY_PARAMETER: &str = "invalid_query_parameter";

fn invalid_query_parameter(message: String) -> Response {
    error::bad_request(INVALID_QUERY_PARAMETER, message)
}

/// The query of `GET /v1/responses/{id}`: `include` (the Responses API's
/// includable fields, the set the conversation item routes take; the stored
/// object already carries everything the gateway kept of the response, so a
/// valid value changes nothing) and `stream`. The gateway keeps no event
/// stream to join for a stored response, so `stream=true` is refused by name
/// instead of being answered with a JSON object a stream client cannot read;
/// `stream=false` and an absent `stream` read the object. Parameters the
/// contract does not name are ignored.
pub fn parse_response_read_query(query: Option<&str>) -> Result<(), Response> {
    parse_item_include_query(query)?;
    let streams = url::form_urlencoded::parse(query.unwrap_or("").as_bytes())
        .filter(|(key, _)| key == "stream");
    for (_, value) in streams {
        match &*value {
            "false" => {}
            "true" => {
                return Err(invalid_query_parameter(
                    "stream: a stored response is read as one JSON object on this route and \
                     has no event stream to join; omit stream or send stream=false"
                        .to_string(),
                ))
            }
            _ => {
                return Err(invalid_query_parameter(format!(
                    "stream: Input should be 'true' or 'false', got {value:?}"
                )))
            }
        }
    }
    Ok(())
}

/// The query of `GET /v1/responses/{id}/input_items`: `limit` (1 to 100, 20 by
/// default), `order` (`asc`, the order the request gave the items, or `desc`,
/// the default) and `after` (the id of an item; the page is the one after it
/// in that order; an id the list does not hold starts the page at its first
/// item, as the conversation item list answers such a cursor). `include` is
/// validated as on the object read. Parameters the contract does not name
/// are ignored.
#[derive(Debug)]
pub struct InputItemsQuery {
    limit: usize,
    order: SortOrder,
    after: Option<String>,
}

impl InputItemsQuery {
    /// The page size when the query names none.
    pub const DEFAULT_LIMIT: usize = 20;
    /// The largest page the public API serves.
    pub const MAX_LIMIT: usize = 100;

    /// The query of a request's URI (`None` or empty: the defaults); a value
    /// the contract does not take is refused with the parameter named, an
    /// `include` value as the conversation item routes refuse one.
    pub fn parse(query: Option<&str>) -> Result<Self, Response> {
        parse_item_include_query(query)?;
        let mut parsed = Self {
            limit: Self::DEFAULT_LIMIT,
            order: SortOrder::Desc,
            after: None,
        };
        for (key, value) in url::form_urlencoded::parse(query.unwrap_or("").as_bytes()) {
            match &*key {
                "limit" => {
                    parsed.limit = value
                        .parse::<usize>()
                        .ok()
                        .filter(|limit| (1..=Self::MAX_LIMIT).contains(limit))
                        .ok_or_else(|| {
                            invalid_query_parameter(format!(
                                "limit: Input should be an integer from 1 to {}, got {value:?}",
                                Self::MAX_LIMIT
                            ))
                        })?;
                }
                "order" => {
                    parsed.order = match &*value {
                        "asc" => SortOrder::Asc,
                        "desc" => SortOrder::Desc,
                        _ => {
                            return Err(invalid_query_parameter(format!(
                                "order: Input should be 'asc' or 'desc', got {value:?}"
                            )))
                        }
                    }
                }
                "after" => parsed.after = Some(value.into_owned()),
                _ => {}
            }
        }
        Ok(parsed)
    }

    /// The page of `items` (given in the order the request gave them) this
    /// query asks for, and whether items follow it.
    fn page(&self, mut items: Vec<Value>) -> (Vec<Value>, bool) {
        if self.order == SortOrder::Desc {
            items.reverse();
        }
        let first = self
            .after
            .as_deref()
            .and_then(|after| items.iter().position(|item| item["id"] == after))
            .map_or(0, |at| at + 1);
        let has_more = items.len() > first + self.limit;
        let page = items.into_iter().skip(first).take(self.limit).collect();
        (page, has_more)
    }
}

/// `GET /v1/responses/{id}` for a caller of `tenant`, under `query` (see
/// [`parse_response_read_query`]).
pub async fn get_response(
    response_storage: &Arc<dyn ResponseStorage>,
    tenant: Option<&str>,
    response_id: &str,
    query: Option<&str>,
) -> Response {
    if let Err(refused) = parse_response_read_query(query) {
        return refused;
    }
    let id = ResponseId::from(response_id);
    match visible_response(response_storage, &id, tenant).await {
        Ok(Some(stored)) => (StatusCode::OK, Json(stored.raw_response)).into_response(),
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => error::storage_failure("storage_error", "Failed to get response", &e),
    }
}

/// `DELETE /v1/responses/{id}` for a caller of `tenant`.
pub async fn delete_response(
    response_storage: &Arc<dyn ResponseStorage>,
    tenant: Option<&str>,
    response_id: &str,
) -> Response {
    let id = ResponseId::from(response_id);

    match visible_response(response_storage, &id, tenant).await {
        Ok(Some(_)) => match response_storage.delete_response(&id).await {
            Ok(()) => {
                info!(response_id = %id.0, "Deleted response");
                (
                    StatusCode::OK,
                    Json(json!({
                        "id": id.0,
                        "object": "response.deleted",
                        "deleted": true,
                    })),
                )
                    .into_response()
            }
            Err(e) => error::storage_failure("storage_error", "Failed to delete response", &e),
        },
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => error::storage_failure("storage_error", "Failed to get response", &e),
    }
}

/// `GET /v1/responses/{id}/input_items` for a caller of `tenant`: the page
/// of the stored input that `query` asks for (see [`InputItemsQuery`]). An
/// item stored without an id or with an empty one (a record from before ids were minted at store
/// time) is listed under the id `listing_id` derives from the response id and
/// the item's position, the same on every read, so a page's `last_id` is
/// found by the next request whichever item it names; a cursor the list does
/// not hold starts the page at its first item, as the conversation item list
/// answers such a cursor.
pub async fn list_response_input_items(
    response_storage: &Arc<dyn ResponseStorage>,
    tenant: Option<&str>,
    response_id: &str,
    query: Option<&str>,
) -> Response {
    let page = match InputItemsQuery::parse(query) {
        Ok(page) => page,
        Err(refused) => return refused,
    };
    let resp_id = ResponseId::from(response_id);

    match visible_response(response_storage, &resp_id, tenant).await {
        Ok(Some(stored)) => {
            let items = stored.input.as_array().cloned().unwrap_or_default();

            let items_with_ids: Vec<Value> = items
                .into_iter()
                .enumerate()
                .map(|(position, mut item)| {
                    if item
                        .get("id")
                        .and_then(Value::as_str)
                        .is_none_or(str::is_empty)
                    {
                        if let Some(obj) = item.as_object_mut() {
                            obj.insert("id".to_string(), json!(listing_id(&resp_id, position)));
                        }
                    }
                    item
                })
                .collect();
            let (data, has_more) = page.page(items_with_ids);
            let id_of = |item: &Value| item.get("id").and_then(Value::as_str).map(str::to_owned);
            let first_id = data.first().and_then(id_of);
            let last_id = data.last().and_then(id_of);

            let response_body = json!({
                "object": "list",
                "data": data,
                "first_id": first_id,
                "last_id": last_id,
                "has_more": has_more
            });

            (StatusCode::OK, Json(response_body)).into_response()
        }
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => {
            warn!("Failed to retrieve input items for {}: {}", response_id, e);
            error::storage_failure("storage_error", "Failed to retrieve input items", &e)
        }
    }
}

/// The listing id of the item at `position` of a stored response's input
/// when the item was stored without one: the response id's body under the
/// `msg` prefix the store mints, with the position, so every read of the
/// record answers the same id and a page's `last_id` is found by the next
/// request.
fn listing_id(response_id: &ResponseId, position: usize) -> String {
    let body = response_id
        .0
        .strip_prefix("resp_")
        .unwrap_or(&response_id.0);
    format!("msg_{body}_{position}")
}

#[cfg(test)]
#[path = "../../../tests/unit/endpoints/responses/handlers/tests.rs"]
mod tests;
