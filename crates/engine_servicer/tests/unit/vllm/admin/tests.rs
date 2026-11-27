use tonic::metadata::MetadataValue;

use super::*;

#[test]
fn grpc_timeout_follows_the_header_grammar() {
    let parse = |value: &str| {
        let mut metadata = MetadataMap::new();
        metadata.insert("grpc-timeout", MetadataValue::try_from(value).unwrap());
        grpc_timeout(&metadata)
    };
    assert_eq!(parse("2H"), Some(Duration::from_secs(7200)));
    assert_eq!(parse("3M"), Some(Duration::from_secs(180)));
    assert_eq!(parse("30S"), Some(Duration::from_secs(30)));
    assert_eq!(parse("250m"), Some(Duration::from_millis(250)));
    assert_eq!(parse("7u"), Some(Duration::from_micros(7)));
    assert_eq!(parse("9n"), Some(Duration::from_nanos(9)));
    for malformed in ["", "S", "5", "123456789S", "+5S", "5x"] {
        assert_eq!(parse(malformed), None, "{malformed:?}");
    }
    assert_eq!(grpc_timeout(&MetadataMap::new()), None);
}
