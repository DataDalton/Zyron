//! Live upgrade endpoints.
//!
//! Four paths report what the substrate is doing, as JSON on a plain GET and
//! as a WebSocket subscription that pushes a new document whenever the
//! answer changes. The documents are the same either way, so a dashboard and
//! a `curl` see one shape.
//!
//! A subscription sends only when something moved. The board and the
//! migration progress counters are both cheap to read, so the poll compares
//! a fingerprint of the rendered document rather than holding a change
//! notification, which keeps the endpoint free of any coupling to the
//! orchestrator

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use zyron_common::format::FormatSubstrate;

/// Live upgrade state per node
pub const STATE_PATH: &str = "/api/upgrade/state";
/// Per-format migration progress
pub const FORMAT_MIGRATIONS_PATH: &str = "/api/upgrade/format_migrations";
/// The user-object rewrite queue
pub const REWRITES_PATH: &str = "/api/upgrade/rewrites";
/// Deprecation warnings emitted in the trailing window
pub const DEPRECATION_WARNINGS_PATH: &str = "/api/upgrade/deprecation_warnings";

/// Every path this module serves
pub const PATHS: &[&str] = &[
    STATE_PATH,
    FORMAT_MIGRATIONS_PATH,
    REWRITES_PATH,
    DEPRECATION_WARNINGS_PATH,
];

/// How long a subscription waits between checks for a change
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// How long a quiet subscription waits before sending a keepalive, so a
/// proxy in the path does not close an idle connection
const KEEPALIVE_AFTER: Duration = Duration::from_secs(20);

/// Whether a path is one of these endpoints
pub fn is_upgrade_path(path: &str) -> bool {
    PATHS.contains(&path)
}

/// Renders one endpoint's document.
///
/// An unknown path renders an error document rather than nothing, so a
/// caller always gets JSON back
pub fn render(path: &str, now_secs: u64) -> String {
    let Ok(substrate) = zyron_common::format::substrate() else {
        return r#"{"error":"the format substrate is not loaded"}"#.to_string();
    };
    match path {
        STATE_PATH => render_state(),
        FORMAT_MIGRATIONS_PATH => render_format_migrations(substrate, now_secs),
        REWRITES_PATH => render_rewrites(),
        DEPRECATION_WARNINGS_PATH => render_deprecation_warnings(substrate),
        other => format!(
            r#"{{"error":"unknown upgrade endpoint {}"}}"#,
            escape(other)
        ),
    }
}

fn render_state() -> String {
    let board = zyron_common::format::upgrade_board();
    let settings = board.settings();
    let mut nodes = String::from("[");
    for (index, state) in board.node_states().into_iter().enumerate() {
        if index > 0 {
            nodes.push(',');
        }
        nodes.push_str(&format!(
            r#"{{"node_id":"{}","phase":"{}","from_version":"{}","to_version":"{}","is_leader":{},"started_at_secs":{},"updated_at_secs":{},"message":"{}"}}"#,
            escape(&state.node_id),
            state.phase.label(),
            escape(&state.from_version),
            escape(&state.to_version),
            state.is_leader,
            state.started_at_secs,
            state.updated_at_secs,
            escape(&state.message)
        ));
    }
    nodes.push(']');
    format!(
        r#"{{"cluster_phase":"{}","channel":"{}","auto_upgrade_enabled":{},"paused":{},"window":"{}","nodes":{}}}"#,
        board.cluster_phase().label(),
        escape(settings.channel.label()),
        settings.auto_upgrade_enabled,
        settings.paused,
        escape(&settings.window.to_string()),
        nodes
    )
}

fn render_format_migrations(substrate: &FormatSubstrate, now_secs: u64) -> String {
    let mut out = String::from(r#"{"migrations":["#);
    for (index, run) in substrate.migrations.runs().into_iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let eta = run
            .eta_secs(now_secs)
            .map(|secs| secs.to_string())
            .unwrap_or_else(|| "null".to_string());
        out.push_str(&format!(
            r#"{{"format_kind":"{}","from_version":"{}","to_version":"{}","policy":"{}","files_total":{},"files_done":{},"bytes_remaining":{},"percent_complete":{:.1},"eta_secs":{},"failures":{},"paused":{},"finished":{}}}"#,
            run.kind.catalog_name(),
            run.from,
            run.to,
            run.policy.label(),
            run.files_total(),
            run.files_done(),
            run.bytes_remaining(),
            run.percent_complete(),
            eta,
            run.failures(),
            run.is_paused(),
            run.is_finished()
        ));
    }
    out.push_str("]}");
    out
}

fn render_rewrites() -> String {
    let records = zyron_common::format::upgrade_board().rewrites();
    let mut out = String::from(r#"{"rewrites":["#);
    for (index, record) in records.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            r#"{{"object_name":"{}","object_kind":"{}","rewriter":"{}","category":"{}","status":"{}","before_hash":{},"after_hash":{},"acknowledged_by":"{}","updated_at_secs":{}}}"#,
            escape(&record.object_name),
            record.object_kind.catalog_name(),
            escape(&record.rewriter_name),
            record.category.label(),
            record.status.label(),
            record.before_hash,
            record.after_hash,
            escape(&record.acknowledged_by),
            record.updated_at_secs
        ));
    }
    out.push_str("]}");
    out
}

fn render_deprecation_warnings(substrate: &FormatSubstrate) -> String {
    let rows = substrate.warning_log.report(0);
    let mut out = format!(
        r#"{{"rate_limit_per_hour":{},"emitted_total":{},"suppressed_total":{},"items":["#,
        substrate.warning_limiter.limit(),
        substrate.warning_limiter.emitted(),
        substrate.warning_limiter.suppressed()
    );
    for (index, row) in rows.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let tenants: Vec<String> = row
            .tenants
            .iter()
            .map(|tenant| format!("\"{}\"", escape(tenant)))
            .collect();
        out.push_str(&format!(
            r#"{{"item_id":"{}","item_kind":"{}","uses_warned":{},"tenants":[{}]}}"#,
            escape(&row.item_id),
            row.item_kind.label(),
            row.count,
            tenants.join(",")
        ));
    }
    out.push_str("]}");
    out
}

/// Escapes a string for a JSON document
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Whether the upgrade is in a phase worth waking a dashboard for, which the
/// subscription uses to decide its keepalive cadence
pub fn is_active() -> bool {
    !zyron_common::format::upgrade_board()
        .cluster_phase()
        .is_terminal()
}

/// Serves one path as a WebSocket subscription.
///
/// Returns whether the handshake was taken. A request with no upgrade header
/// falls back to the plain GET, which the HTTP router answers
pub async fn serve_stream<S>(
    stream: &mut S,
    request: &crate::gateway::request::HttpRequest,
    path: &str,
    shutdown: Arc<AtomicBool>,
) -> bool
where
    S: AsyncReadExt + AsyncWriteExt + Unpin,
{
    let Some(accept) = crate::gateway::websocket::upgrade_accept(request) else {
        return false;
    };
    if stream
        .write_all(&crate::gateway::websocket::upgrade_response(&accept))
        .await
        .is_err()
    {
        return true;
    }

    let mut connection = crate::gateway::websocket::WebSocketConnection::new();
    let mut read_buf = vec![0u8; 1024];
    let mut last_sent: Option<String> = None;
    let mut quiet_for = Duration::ZERO;

    loop {
        if shutdown.load(Ordering::Acquire) || connection.is_closed() {
            return true;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let document = render(path, now);
        let changed = last_sent.as_deref() != Some(document.as_str());
        if changed || quiet_for >= KEEPALIVE_AFTER {
            let frame = crate::gateway::websocket::encode_frame(
                crate::gateway::websocket::WsOpcode::Text,
                document.as_bytes(),
                true,
            );
            if stream.write_all(&frame).await.is_err() {
                return true;
            }
            last_sent = Some(document);
            quiet_for = Duration::ZERO;
        } else {
            quiet_for += POLL_INTERVAL;
        }

        // A client close or ping arrives on the same socket, so the wait is
        // a read with a timeout rather than a sleep
        match tokio::time::timeout(POLL_INTERVAL, stream.read(&mut read_buf)).await {
            Ok(Ok(0)) => return true,
            Ok(Ok(read)) => {
                let (_, messages) = connection.absorb(&read_buf[..read]);
                for message in messages {
                    if let crate::gateway::websocket::WsMessage::Ping(payload) = message {
                        let pong = crate::gateway::websocket::encode_frame(
                            crate::gateway::websocket::WsOpcode::Pong,
                            &payload,
                            true,
                        );
                        if stream.write_all(&pong).await.is_err() {
                            return true;
                        }
                    }
                }
            }
            Ok(Err(_)) => return true,
            Err(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::rewrite::{ObjectKind, RewriteCategory, RewriteStatus};
    use zyron_common::format::{NodeUpgradeState, RewriteRecord, UpgradePhase};

    #[test]
    fn test_every_path_renders_json() {
        for path in PATHS {
            let document = render(path, 1_000);
            assert!(document.starts_with('{'), "{path} rendered {document}");
            assert!(document.ends_with('}'), "{path} rendered {document}");
            assert!(
                !document.contains("\"error\""),
                "{path} rendered {document}"
            );
        }
    }

    #[test]
    fn test_an_unknown_path_renders_an_error_document() {
        let document = render("/api/upgrade/nothing", 0);
        assert!(document.contains("unknown upgrade endpoint"), "{document}");
    }

    #[test]
    fn test_is_upgrade_path_covers_exactly_the_four() {
        assert_eq!(PATHS.len(), 4);
        for path in PATHS {
            assert!(is_upgrade_path(path));
        }
        assert!(!is_upgrade_path("/api/upgrade"));
        assert!(!is_upgrade_path("/pressure"));
    }

    #[test]
    fn test_the_state_document_reports_the_nodes_on_the_board() {
        let board = zyron_common::format::upgrade_board();
        board.set_node_state(NodeUpgradeState {
            node_id: "node-1".to_string(),
            from_version: "0.11.0".to_string(),
            to_version: "0.12.0".to_string(),
            phase: UpgradePhase::Rolling,
            started_at_secs: 1,
            updated_at_secs: 2,
            is_leader: true,
            message: "restarting on the new binary".to_string(),
        });
        let document = render(STATE_PATH, 0);
        assert!(document.contains("\"node_id\":\"node-1\""), "{document}");
        assert!(document.contains("\"phase\":\"Rolling\""), "{document}");
        assert!(document.contains("\"is_leader\":true"), "{document}");
        assert!(is_active());
    }

    #[test]
    fn test_the_rewrite_document_reports_the_queue() {
        let board = zyron_common::format::upgrade_board();
        board.set_rewrites(vec![RewriteRecord {
            object_name: "sales_view".to_string(),
            object_kind: ObjectKind::View,
            rewriter_name: "warehouse_to_compute".to_string(),
            category: RewriteCategory::Safe,
            status: RewriteStatus::Applied,
            before_hash: 11,
            after_hash: 22,
            acknowledged_by: String::new(),
            updated_at_secs: 5,
            diff: "- a\n+ b".to_string(),
        }]);
        let document = render(REWRITES_PATH, 0);
        assert!(
            document.contains("\"object_name\":\"sales_view\""),
            "{document}"
        );
        assert!(document.contains("\"category\":\"safe\""), "{document}");
        assert!(document.contains("\"status\":\"applied\""), "{document}");
    }

    #[test]
    fn test_a_quote_in_a_message_is_escaped() {
        let board = zyron_common::format::upgrade_board();
        board.set_node_state(NodeUpgradeState {
            node_id: "node-\"quoted\"".to_string(),
            from_version: "0.11.0".to_string(),
            to_version: "0.12.0".to_string(),
            phase: UpgradePhase::Paused,
            started_at_secs: 0,
            updated_at_secs: 0,
            is_leader: false,
            message: "a line\nand another".to_string(),
        });
        let document = render(STATE_PATH, 0);
        assert!(document.contains("node-\\\"quoted\\\""), "{document}");
        assert!(document.contains("a line\\nand another"), "{document}");
        assert!(!document.contains('\n'), "the document is one line");
    }

    #[test]
    fn test_escaping_covers_control_characters() {
        assert_eq!(escape("plain"), "plain");
        assert_eq!(escape("a\"b"), "a\\\"b");
        assert_eq!(escape("a\\b"), "a\\\\b");
        assert_eq!(escape("a\tb"), "a\\tb");
        assert_eq!(escape("a\u{1}b"), "a\\u0001b");
    }
}
