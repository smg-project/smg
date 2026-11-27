use super::*;
use crate::inventory::ToolEntry;

fn test_engine() -> PolicyEngine {
    PolicyEngine::new(Arc::new(AuditLog::new()))
}

#[test]
fn test_default_policy() {
    let engine = test_engine().with_default_policy(PolicyDecision::Allow);
    let tenant = TenantContext::new("test");
    let hints = ToolAnnotations::new();

    let decision = engine.evaluate("server", "tool", &hints, &tenant, "req-1");
    assert!(decision.is_allowed());
}

#[test]
fn a_tool_listed_with_a_title_alone_keeps_the_default_decision() {
    // The engine reads the entry's approval set, which a listing does not
    // fill: a tool whose server declared no behaviour hint (a title alone)
    // gets the default decision, not the refusal the spec's default for an
    // absent destructive hint would mean. `from_yaml_config` registers no
    // rule either, so this is the decision without an explicit policy.
    use rmcp::model::{Tool, ToolAnnotations as RmcpToolAnnotations};

    let mut tool = Tool::new(
        "create_issue",
        "Create an issue",
        Arc::new(serde_json::Map::new()),
    );
    tool.annotations = Some(RmcpToolAnnotations::with_title("Create issue"));
    let entry = ToolEntry::from_server_tool("issues", tool);
    let decision = test_engine().evaluate(
        "issues",
        "create_issue",
        &entry.annotations,
        &TenantContext::new("test"),
        "req-1",
    );
    assert!(decision.is_allowed());
}

#[test]
fn test_tool_policy() {
    let engine = test_engine().with_tool_policy(
        QualifiedToolName::new("server", "dangerous_tool"),
        ToolPolicy {
            decision: PolicyDecision::Deny,
        },
    );

    let tenant = TenantContext::new("test");
    let hints = ToolAnnotations::new();

    let decision = engine.evaluate("server", "dangerous_tool", &hints, &tenant, "req-1");
    assert!(!decision.is_allowed());
}

#[test]
fn test_server_trust_level() {
    let engine = test_engine().with_server_policy(
        "trusted_server",
        ServerPolicy {
            default: PolicyDecision::Allow,
            trust_level: TrustLevel::Trusted,
        },
    );

    let tenant = TenantContext::new("test");
    let hints = ToolAnnotations::new().with_destructive(true);

    let decision = engine.evaluate("trusted_server", "tool", &hints, &tenant, "req-1");
    assert!(decision.is_allowed());
}

#[test]
fn test_untrusted_server() {
    let engine = test_engine().with_server_policy(
        "untrusted",
        ServerPolicy {
            default: PolicyDecision::Allow,
            trust_level: TrustLevel::Untrusted,
        },
    );

    let tenant = TenantContext::new("test");
    let destructive = ToolAnnotations::new().with_destructive(true);
    let read_only = ToolAnnotations::new().with_read_only(true);

    let decision = engine.evaluate("untrusted", "tool", &destructive, &tenant, "req-1");
    assert!(!decision.is_allowed());

    let decision = engine.evaluate("untrusted", "tool", &read_only, &tenant, "req-2");
    assert!(decision.is_allowed());
}

#[test]
fn test_sandboxed_server() {
    let engine = test_engine().with_server_policy(
        "sandbox",
        ServerPolicy {
            default: PolicyDecision::Allow,
            trust_level: TrustLevel::Sandboxed,
        },
    );

    let tenant = TenantContext::new("test");
    let open_world = ToolAnnotations::new().with_open_world(true);
    let read_only = ToolAnnotations::new().with_read_only(true);

    let decision = engine.evaluate("sandbox", "tool", &open_world, &tenant, "req-1");
    assert!(!decision.is_allowed());

    let decision = engine.evaluate("sandbox", "tool", &read_only, &tenant, "req-2");
    assert!(decision.is_allowed());
}

#[test]
fn test_pattern_rule() {
    let engine = test_engine()
        .with_default_policy(PolicyDecision::Allow)
        .with_rule(PolicyRule::new(
            "block_delete",
            RulePattern::Tool(Regex::new("^delete_").unwrap()),
            RuleCondition::Always,
            PolicyDecision::Deny,
        ));

    let tenant = TenantContext::new("test");
    let hints = ToolAnnotations::new();

    let decision = engine.evaluate("server", "delete_user", &hints, &tenant, "req-1");
    assert!(!decision.is_allowed());

    let decision = engine.evaluate("server", "get_user", &hints, &tenant, "req-2");
    assert!(decision.is_allowed());
}

#[test]
fn test_annotation_based_decision() {
    let engine = test_engine().with_default_policy(PolicyDecision::Deny);
    let tenant = TenantContext::new("test");

    let read_only = ToolAnnotations::new().with_read_only(true);
    let decision = engine.evaluate("server", "tool", &read_only, &tenant, "req-1");
    assert!(decision.is_allowed());
}
