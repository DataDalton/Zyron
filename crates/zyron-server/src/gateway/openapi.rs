// -----------------------------------------------------------------------------
// OpenAPI 3.0.3 emission.
//
// Walks the catalog endpoint list and hand-builds a JSON spec. Path parameters
// named with :name are rewritten to {name} so consumers can render the spec in
// any OpenAPI tool. Emits a compact Swagger-style HTML page for browser
// exploration without external scripts.
// -----------------------------------------------------------------------------

use std::sync::Arc;

use zyron_catalog::schema::{EndpointAuthMode, EndpointEntry, HttpMethod as CatalogMethod};

/// Emits a full OpenAPI JSON document describing every endpoint in the list.
pub fn emit_openapi_json(endpoints: &[Arc<EndpointEntry>]) -> String {
    let mut out = String::with_capacity(2048);
    out.push('{');
    write_kv(&mut out, "openapi", "\"3.0.3\"", false);
    out.push(',');
    out.push_str("\"info\":{");
    write_kv(&mut out, "title", "\"ZyronDB Endpoints\"", false);
    out.push(',');
    write_kv(&mut out, "version", "\"1.0\"", false);
    out.push('}');
    out.push(',');
    out.push_str("\"paths\":{");
    let mut first_path = true;
    for entry in endpoints {
        if !first_path {
            out.push(',');
        }
        first_path = false;
        write_path(&mut out, entry);
    }
    out.push('}');
    out.push(',');
    out.push_str("\"components\":{");
    out.push_str("\"securitySchemes\":{");
    out.push_str("\"bearerAuth\":{\"type\":\"http\",\"scheme\":\"bearer\"},");
    out.push_str("\"apiKey\":{\"type\":\"apiKey\",\"name\":\"X-API-Key\",\"in\":\"header\"},");
    out.push_str("\"basicAuth\":{\"type\":\"http\",\"scheme\":\"basic\"}");
    out.push('}');
    out.push('}');
    out.push('}');
    out
}

fn write_path(out: &mut String, entry: &EndpointEntry) {
    let normalized = normalize_path(&entry.path);
    out.push('"');
    escape_into(out, &normalized);
    out.push('"');
    out.push(':');
    out.push('{');
    let mut first_method = true;
    for method in &entry.methods {
        if !first_method {
            out.push(',');
        }
        first_method = false;
        let key = method_key(*method);
        out.push('"');
        out.push_str(key);
        out.push_str("\":{");
        write_kv(
            out,
            "operationId",
            &format!("\"{}_{}\"", entry.name, key),
            false,
        );
        out.push(',');
        out.push_str("\"parameters\":[");
        let mut first_param = true;
        for name in extract_path_params(&entry.path) {
            if !first_param {
                out.push(',');
            }
            first_param = false;
            out.push_str("{\"name\":\"");
            escape_into(out, &name);
            out.push_str("\",\"in\":\"path\",\"required\":true,\"schema\":{\"type\":\"string\"}}");
        }
        out.push(']');
        out.push(',');
        out.push_str("\"responses\":{\"200\":{\"description\":\"success\"}}");
        if !matches!(entry.auth_mode, EndpointAuthMode::None) {
            out.push(',');
            out.push_str("\"security\":[");
            let scheme = match entry.auth_mode {
                EndpointAuthMode::Jwt | EndpointAuthMode::OAuth2 => "bearerAuth",
                EndpointAuthMode::ApiKey => "apiKey",
                EndpointAuthMode::Basic => "basicAuth",
                EndpointAuthMode::Mtls => "bearerAuth",
                EndpointAuthMode::None => "",
            };
            out.push_str("{\"");
            out.push_str(scheme);
            out.push_str("\":[]}]");
        }
        out.push('}');
    }
    out.push('}');
}

fn normalize_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for seg in path.split('/') {
        if seg.is_empty() {
            continue;
        }
        out.push('/');
        if let Some(rest) = seg.strip_prefix(':') {
            out.push('{');
            out.push_str(rest);
            out.push('}');
        } else {
            out.push_str(seg);
        }
    }
    if out.is_empty() {
        out.push('/');
    }
    out
}

fn extract_path_params(path: &str) -> Vec<String> {
    let mut names = Vec::new();
    for seg in path.split('/') {
        if let Some(rest) = seg.strip_prefix(':') {
            names.push(rest.to_string());
        } else if seg.starts_with('{') && seg.ends_with('}') && seg.len() >= 2 {
            names.push(seg[1..seg.len() - 1].to_string());
        }
    }
    names
}

fn method_key(m: CatalogMethod) -> &'static str {
    match m {
        CatalogMethod::Get => "get",
        CatalogMethod::Post => "post",
        CatalogMethod::Put => "put",
        CatalogMethod::Delete => "delete",
        CatalogMethod::Patch => "patch",
        CatalogMethod::Head => "head",
        CatalogMethod::Options => "options",
    }
}

fn write_kv(out: &mut String, key: &str, raw_value: &str, quote_value: bool) {
    out.push('"');
    out.push_str(key);
    out.push('"');
    out.push(':');
    if quote_value {
        out.push('"');
        escape_into(out, raw_value);
        out.push('"');
    } else {
        out.push_str(raw_value);
    }
}

fn escape_into(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = std::fmt::Write::write_fmt(out, format_args!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

/// Minimal HTML page rendering the endpoint catalog. No external scripts.
pub fn emit_swagger_html(endpoints: &[Arc<EndpointEntry>]) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str(
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>ZyronDB Endpoints</title>\
         <style>body{font-family:sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}\
         .ep{border:1px solid #ddd;border-radius:6px;padding:1rem;margin-bottom:1rem}\
         .m{display:inline-block;padding:2px 6px;border-radius:4px;color:#fff;font-size:12px;\
         margin-right:6px}.get{background:#16a34a}.post{background:#2563eb}.put{background:#ca8a04}\
         .delete{background:#dc2626}.patch{background:#7c3aed}.options{background:#64748b}\
         .head{background:#0891b2}.path{font-family:monospace;font-size:14px}\
         .meta{color:#475569;font-size:12px;margin-top:4px}</style></head><body>",
    );
    out.push_str("<h1>ZyronDB Endpoints</h1>");
    out.push_str(&format!(
        "<p>{} endpoint(s) registered.</p>",
        endpoints.len()
    ));
    for entry in endpoints {
        out.push_str("<div class=\"ep\">");
        for m in &entry.methods {
            let key = method_key(*m);
            out.push_str(&format!(
                "<span class=\"m {}\">{}</span>",
                key,
                key.to_uppercase()
            ));
        }
        out.push_str("<span class=\"path\">");
        out.push_str(&html_escape(&entry.path));
        out.push_str("</span>");
        out.push_str("<div class=\"meta\">name=");
        out.push_str(&html_escape(&entry.name));
        out.push_str(&format!(
            " auth={:?} scopes={} enabled={}",
            entry.auth_mode,
            entry.required_scopes.len(),
            entry.enabled
        ));
        out.push_str("</div>");
        out.push_str("</div>");
    }
    out.push_str("</body></html>");
    out
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_catalog::schema::{EndpointEntry, EndpointKind, HttpMethod};
    use zyron_catalog::{EndpointId, SchemaId};

    fn sample(name: &str, path: &str, methods: Vec<HttpMethod>) -> Arc<EndpointEntry> {
        Arc::new(EndpointEntry {
            id: EndpointId(1),
            schema_id: SchemaId(1),
            name: name.into(),
            kind: EndpointKind::Rest,
            path: path.into(),
            methods,
            sql_body: "SELECT 1".into(),
            backed_publication_id: None,
            auth_mode: EndpointAuthMode::Jwt,
            required_scopes: vec!["read".into()],
            output_format: None,
            cors_origins: Vec::new(),
            rate_limit: None,
            cache_seconds: None,
            timeout_seconds: None,
            max_request_body_kb: None,
            message_format: None,
            heartbeat_seconds: None,
            backpressure: None,
            max_connections: None,
            enabled: true,
            owner_role_id: 0,
            created_at: 0,
        })
    }

    #[test]
    fn emits_openapi_version() {
        let eps = vec![sample("a", "/api/a", vec![HttpMethod::Get])];
        let json = emit_openapi_json(&eps);
        assert!(json.starts_with("{\"openapi\":\"3.0.3\""));
    }

    #[test]
    fn path_param_rewritten_to_curly() {
        let eps = vec![sample("a", "/api/:id", vec![HttpMethod::Get])];
        let json = emit_openapi_json(&eps);
        assert!(json.contains("\"/api/{id}\""));
        assert!(json.contains("\"name\":\"id\""));
    }

    #[test]
    fn emits_security_scheme_for_jwt() {
        let eps = vec![sample("a", "/api/a", vec![HttpMethod::Get])];
        let json = emit_openapi_json(&eps);
        assert!(json.contains("bearerAuth"));
        assert!(json.contains("\"security\""));
    }

    #[test]
    fn html_lists_every_endpoint() {
        let eps = vec![
            sample("a", "/api/a", vec![HttpMethod::Get]),
            sample("b", "/api/b", vec![HttpMethod::Post]),
        ];
        let html = emit_swagger_html(&eps);
        assert!(html.contains("/api/a"));
        assert!(html.contains("/api/b"));
        assert!(html.contains("2 endpoint(s)"));
    }

    #[test]
    fn normalize_path_handles_mixed_styles() {
        assert_eq!(normalize_path("/api/:x/:y"), "/api/{x}/{y}");
        assert_eq!(normalize_path("/api/{x}"), "/api/{x}");
    }
}
