use super::*;
use crate::tenant::TenantKey;

/// Two iterations of one request reserve as the request's tenant under
/// charge ids of their own, distinct from each other and from the
/// request's.
#[test]
fn each_iteration_reserves_under_its_own_charge_id() {
    let request = TenantRequestMeta::new(TenantKey::new("auth:tenant"));
    let first = iteration_charge(&request);
    let second = iteration_charge(&request);

    assert_eq!(first.tenant_key, request.tenant_key);
    assert_eq!(second.tenant_key, request.tenant_key);
    assert_ne!(first.request_charge_id, request.request_charge_id);
    assert_ne!(second.request_charge_id, request.request_charge_id);
    assert_ne!(first.request_charge_id, second.request_charge_id);
}
