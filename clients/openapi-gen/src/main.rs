//! `openapi-gen`: writes the gateway's OpenAPI document as YAML to the path
//! given (`-` for stdout; the clients' default path otherwise).

use std::io::Write;

fn main() -> anyhow::Result<()> {
    let doc = openapi_gen::document()?;

    // Output path: first CLI arg or default
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "clients/openapi/smg-openapi.yaml".to_string());

    let yaml = serde_yaml_ng::to_string(&doc)?;

    if output_path == "-" {
        write!(std::io::stdout(), "{yaml}")?;
    } else {
        std::fs::write(&output_path, &yaml)?;
        writeln!(std::io::stderr(), "Wrote OpenAPI spec to {output_path}")?;
    }

    Ok(())
}
