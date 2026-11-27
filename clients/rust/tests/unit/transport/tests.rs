use super::query_url;

#[test]
fn the_query_url_names_each_pair_in_order_and_encodes_the_values() {
    let url = query_url(
        "http://gateway.test/v1/responses/resp_1/input_items",
        &[
            ("limit", "2".to_string()),
            ("after", "msg 3".to_string()),
            ("include", "a".to_string()),
            ("include", "b".to_string()),
        ],
    )
    .expect("a well-formed url");
    assert_eq!(
        url.as_str(),
        "http://gateway.test/v1/responses/resp_1/input_items?limit=2&after=msg+3&include=a&include=b"
    );
}
