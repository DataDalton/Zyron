//! The `/pressure` endpoint and its live stream.
//!
//! The same signal the SQL views expose, over HTTP, because the things that
//! most need to read it are not SQL clients: a load balancer deciding where to
//! send the next request, an orchestrator deciding whether to add a node, an
//! operator watching a graph.
//!
//! `GET /pressure` answers with the node's current state. `GET
//! /pressure/stream` upgrades to a WebSocket and pushes the same document
//! every controller tick, so a watcher sees a change when it happens rather
//! than at whatever interval it chose to poll.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use zyron_pressure::capability::NodeCapabilities;
use zyron_pressure::pressure_control::PressureController;

/// How often the stream pushes. Matches the controller's own cadence: pushing
/// faster would repeat a document that has not changed, and slower would make
/// the stream a worse answer than polling.
const STREAM_INTERVAL: Duration = Duration::from_millis(100);

/// The HTTP path the snapshot answers on.
pub const PRESSURE_PATH: &str = "/pressure";

/// The HTTP path the live stream upgrades from.
pub const PRESSURE_STREAM_PATH: &str = "/pressure/stream";

/// Where a draining node's working set is fetched from.
///
/// The manifest is the one part of the handover that has to move as bytes, and
/// it moves over the transport the node already serves rather than waiting for
/// a mesh protocol. A survivor asked to take over for a departing node fetches
/// this and prefetches what it names, while the departing node is still
/// serving, so the traffic arrives at a warm pool.
pub const HOT_SET_PATH: &str = "/pressure/hot_set";

/// Escapes a string for a JSON body. Only the characters JSON forbids raw.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
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

/// Renders a float that may be infinite.
///
/// JSON has no infinity, and emitting one produces a document that strict
/// parsers reject. Background's objective is genuinely unbounded, so it is
/// null: absent rather than a number that would compare wrong.
fn number(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "null".to_string()
    }
}

/// The node's pressure as a JSON document.
pub fn render_snapshot(capabilities: Option<&NodeCapabilities>) -> String {
    let controller = PressureController::global();
    let snapshot = controller.snapshot();
    let connections = controller.connections();

    let mut out = String::with_capacity(1024);
    out.push_str("{\"node_id\":");
    out.push_str(&snapshot.node_id.to_string());
    out.push_str(",\"sequence\":");
    out.push_str(&controller.sequence().to_string());
    out.push_str(",\"updated_at_us\":");
    out.push_str(&snapshot.updated_at_us.to_string());
    out.push_str(",\"overall_utilization\":");
    out.push_str(&number(snapshot.overall_utilization));
    out.push_str(",\"bottleneck\":\"");
    out.push_str(snapshot.bottleneck.as_str());
    out.push_str("\",\"actuator\":\"");
    out.push_str(snapshot.actuator_level.as_str());
    out.push_str("\",\"parallel\":{\"total\":");
    out.push_str(&snapshot.parallel_permits_total.to_string());
    out.push_str(",\"available\":");
    out.push_str(&snapshot.parallel_permits_available.to_string());
    out.push_str(",\"scale_pct\":");
    out.push_str(&controller.capacity().scale_pct().to_string());
    out.push_str(",\"base_total\":");
    out.push_str(&controller.capacity().base_total().to_string());
    out.push_str(",\"growth_pct\":");
    out.push_str(&controller.capacity().growth_pct().to_string());
    out.push_str("},\"query_memory_pct\":");
    out.push_str(&controller.query_memory_pct().to_string());
    out.push_str(",\"memory\":{\"reserved_bytes\":");
    out.push_str(&snapshot.memory_reserved_bytes.to_string());
    out.push_str(",\"ceiling_bytes\":");
    out.push_str(&snapshot.memory_ceiling_bytes.to_string());
    out.push_str("},\"connections\":{\"live\":");
    out.push_str(&connections.live().to_string());
    out.push_str(",\"ceiling\":");
    out.push_str(&connections.ceiling().to_string());
    out.push_str(",\"refused_total\":");
    out.push_str(&connections.refused_total().to_string());
    out.push_str("}");

    if let Some(caps) = capabilities {
        out.push_str(",\"hardware\":{\"fingerprint\":\"");
        out.push_str(&caps.fingerprint.to_hex());
        out.push_str("\",\"cpu_model\":\"");
        out.push_str(&escape(&caps.inputs.cpu_model));
        out.push_str("\",\"cores\":");
        out.push_str(&caps.core_count.to_string());
        out.push_str(",\"memory_bytes\":");
        out.push_str(&caps.mem_total_bytes.to_string());
        out.push_str(",\"simd\":\"");
        out.push_str(caps.simd_level.as_str());
        out.push_str("\",\"calibration_inherited\":");
        out.push_str(if caps.inherited { "true" } else { "false" });
        out.push('}');
    }

    out.push_str(",\"classes\":[");
    for (i, row) in snapshot.classes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"class\":\"");
        out.push_str(row.class.as_str());
        out.push_str("\",\"pressure_seconds\":");
        out.push_str(&number(row.pressure_seconds));
        out.push_str(",\"slo_seconds\":");
        out.push_str(&number(row.slo_seconds));
        out.push_str(",\"slo_utilization\":");
        out.push_str(&number(row.slo_utilization()));
        out.push_str(",\"queued_work_seconds\":");
        out.push_str(&number(row.queued_work_seconds));
        out.push_str(",\"active_work_seconds\":");
        out.push_str(&number(row.active_work_seconds));
        out.push_str(",\"queued\":");
        out.push_str(&row.queued_count.to_string());
        out.push_str(",\"in_flight\":");
        out.push_str(&row.in_flight.to_string());
        out.push_str(",\"ceiling\":");
        out.push_str(&row.ceiling.to_string());
        out.push_str(",\"service_capacity_qps\":");
        out.push_str(&number(row.service_capacity_qps));
        out.push_str(",\"calibration_error\":");
        out.push_str(&number(row.calibration_error));
        out.push_str(",\"admitted_total\":");
        out.push_str(&row.admitted_total.to_string());
        out.push_str(",\"shed_total\":");
        out.push_str(&row.shed_total.to_string());

        // Where the class is going, which is what a scale-out is decided on.
        // Carried beside the present reading rather than on a separate path,
        // because a consumer acting on one and not the other would act on half
        // the picture
        let projection = controller.projection(row.class);
        out.push_str(",\"projected\":{\"horizon_seconds\":");
        out.push_str(&number(projection.horizon.as_secs_f64()));
        out.push_str(",\"pressure_seconds\":");
        out.push_str(&number(projection.projected_pressure_seconds));
        out.push_str(",\"breaching\":");
        out.push_str(if projection.breaching_at_horizon() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"arrival_rate_qps\":");
        out.push_str(&number(projection.arrival_rate_qps));
        out.push_str(",\"arrival_rate_derivative\":");
        out.push_str(&number(projection.arrival_rate_derivative));
        out.push_str(",\"arrival_rate_delta\":");
        out.push_str(&number(projection.projected_arrival_rate_delta));
        out.push_str(",\"warm_pool_nodes\":");
        out.push_str(&projection.warm_pool_nodes.to_string());
        out.push_str(",\"warm_pool_uncapped\":");
        out.push_str(&projection.warm_pool_uncapped.to_string());
        out.push_str(",\"trend_trustworthy\":");
        out.push_str(if projection.trend_trustworthy {
            "true"
        } else {
            "false"
        });
        out.push_str("}}");
    }
    out.push(']');

    let registry = zyron_pressure::provisioner::ProvisionerRegistry::global();
    let driver = registry.active();
    let driver_capabilities = driver.capabilities();
    let reach = controller.mesh_reach();
    out.push_str(",\"provisioner\":{\"driver\":\"");
    out.push_str(driver.kind().as_str());
    out.push_str("\",\"can_provision\":");
    out.push_str(if driver_capabilities.can_provision {
        "true"
    } else {
        "false"
    });
    out.push_str(",\"can_reclaim\":");
    out.push_str(if driver_capabilities.can_reclaim {
        "true"
    } else {
        "false"
    });
    out.push_str(",\"max_nodes\":");
    out.push_str(&driver_capabilities.max_nodes.to_string());
    out.push_str(",\"provision_latency_seconds\":");
    out.push_str(&number(controller.provision_horizon().as_secs_f64()));
    out.push_str(",\"completed_provisions\":");
    out.push_str(&registry.completed_provisions(driver.kind()).to_string());
    out.push_str(",\"ladder_can_take_warm_node\":");
    out.push_str(if reach.can_take_warm_node {
        "true"
    } else {
        "false"
    });
    out.push_str(",\"ladder_can_provision\":");
    out.push_str(if reach.can_provision { "true" } else { "false" });
    out.push('}');

    // The classifier's own inputs, so a consumer can see the readings a
    // decision came from rather than only the decision
    let contention = controller.contention();
    out.push_str(",\"contention\":{\"occ_abort_rate\":");
    out.push_str(&number(contention.occ_abort_rate()));
    out.push_str(",\"hot_key_share\":");
    out.push_str(&number(contention.hot_key_share()));
    out.push_str(",\"group_commit_hit_rate\":");
    out.push_str(&number(contention.group_commit_hit_rate()));
    out.push_str(",\"writes_waiting\":");
    out.push_str(&contention.writes_waiting().to_string());
    out.push_str(",\"page_reads\":");
    out.push_str(&contention.page_reads().to_string());
    out.push('}');

    let hot_set = controller.hot_set_status();
    let checkpoint = controller.checkpoint_state();
    out.push_str(",\"hot_set\":{\"pages\":");
    out.push_str(&hot_set.pages.to_string());
    out.push_str(",\"query_shapes\":");
    out.push_str(&hot_set.shapes.to_string());
    out.push_str(",\"generated_us\":");
    out.push_str(&hot_set.generated_us.to_string());
    out.push_str(",\"persisted\":");
    out.push_str(if hot_set.persisted { "true" } else { "false" });
    out.push('}');

    out.push_str(",\"checkpoint\":{\"at_us\":");
    out.push_str(&checkpoint.at_us.to_string());
    out.push_str(",\"age_seconds\":");
    out.push_str(&number(controller.checkpoint_age().as_secs_f64()));
    out.push_str(",\"clean\":");
    out.push_str(if checkpoint.clean { "true" } else { "false" });
    out.push_str(",\"wal_bytes_since\":");
    out.push_str(&checkpoint.wal_bytes_since.to_string());
    out.push('}');

    if let Some(caps) = capabilities {
        // The same call the trigger makes, so a mesh scheduler reading this
        // and a node deciding for itself cannot disagree
        let readiness = crate::background::pressure::scale_to_zero_readiness(controller, caps);
        let estimate = crate::background::pressure::resume_estimate(controller, caps);
        out.push_str(",\"scale_to_zero\":{\"ready\":");
        out.push_str(if readiness.is_ok() { "true" } else { "false" });
        out.push_str(",\"resume_seconds\":");
        out.push_str(&number(estimate.total.as_secs_f64()));
        out.push_str(",\"checkpoint_open_seconds\":");
        out.push_str(&number(estimate.checkpoint_freshness.as_secs_f64()));
        out.push_str(",\"hot_set_reload_seconds\":");
        out.push_str(&number(estimate.hot_set_reload.as_secs_f64()));
        if let Err(e) = &readiness {
            out.push_str(",\"blocked_by\":\"");
            out.push_str(&escape(&e.to_string()));
            out.push('"');
        }
        out.push('}');
    }

    out.push('}');
    out
}

/// Whether a path is one this module serves.
pub fn is_pressure_path(path: &str) -> bool {
    path == PRESSURE_PATH || path == PRESSURE_STREAM_PATH || path == HOT_SET_PATH
}

/// Renders the persisted working-set manifest for a survivor to fetch.
///
/// Base64 rather than raw bytes because the response travels the same JSON
/// path as everything else here, and the manifest is delta encoded before it
/// is base64 encoded, so a million-page working set is still a small response.
///
/// A node with no manifest yet says so rather than returning an empty one. An
/// empty manifest and a missing manifest lead a survivor to opposite actions:
/// one means the departing node was holding nothing, the other means the
/// handover is not ready and moving traffic now would be moving it to a cold
/// pool.
pub fn render_hot_set(data_dir: &std::path::Path) -> (&'static str, String) {
    match zyron_pressure::hot_set::HotSetManifest::load(data_dir) {
        Ok(Some(manifest)) => {
            let encoded = base64_standard(&manifest.encode());
            (
                "200 OK",
                format!(
                    concat!(
                        "{{\"node_id\":{},\"generated_us\":{},\"pages\":{},",
                        "\"query_shapes\":{},\"encoding\":\"base64\",\"manifest\":\"{}\"}}"
                    ),
                    manifest.node_id,
                    manifest.generated_us,
                    manifest.pages.len(),
                    manifest.queries.len(),
                    encoded
                ),
            )
        }
        Ok(None) => (
            "404 Not Found",
            r#"{"error":"this node has not written a working set manifest yet"}"#.into(),
        ),
        Err(e) => (
            "500 Internal Server Error",
            format!(
                "{{\"error\":\"the working set manifest could not be read: {}\"}}",
                escape(&e.to_string())
            ),
        ),
    }
}

/// Standard base64, which is what the manifest field carries.
fn base64_standard(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Reads a manifest back out of a fetched response body.
///
/// The other half of the handover, kept here so the two encodings cannot drift
/// apart: a survivor and the node it is taking over for are running the same
/// build, and a format they disagree about would prefetch the wrong pages
/// while reporting success.
pub fn parse_hot_set(body: &str) -> Option<zyron_pressure::hot_set::HotSetManifest> {
    let field = "\"manifest\":\"";
    let start = body.find(field)? + field.len();
    let rest = &body[start..];
    let end = rest.find('"')?;
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&rest[..end])
        .ok()?;
    zyron_pressure::hot_set::HotSetManifest::decode(&bytes).ok()
}

/// Serves the WebSocket stream on an already-accepted connection.
///
/// Returns without writing anything when the request is not a valid upgrade,
/// so the caller can fall through to answering it as ordinary HTTP.
pub async fn serve_stream<S>(
    stream: &mut S,
    request: &crate::gateway::request::HttpRequest,
    capabilities: Option<Arc<NodeCapabilities>>,
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
    let mut last_sequence = u32::MAX;

    loop {
        if shutdown.load(Ordering::Acquire) || connection.is_closed() {
            return true;
        }

        // A document is only worth sending when the controller has advanced.
        // Resending an unchanged one would make the stream a heartbeat rather
        // than a change feed
        let sequence = PressureController::global().sequence();
        if sequence != last_sequence {
            last_sequence = sequence;
            let body = render_snapshot(capabilities.as_deref());
            let frame = crate::gateway::websocket::encode_frame(
                crate::gateway::websocket::WsOpcode::Text,
                body.as_bytes(),
                true,
            );
            if stream.write_all(&frame).await.is_err() {
                return true;
            }
        }

        // Read with a timeout rather than blocking, so a client that sends
        // nothing still gets pushes and a client that closes is noticed
        match tokio::time::timeout(STREAM_INTERVAL, stream.read(&mut read_buf)).await {
            Ok(Ok(0)) => return true,
            Ok(Ok(n)) => {
                let (_, messages) = connection.absorb(&read_buf[..n]);
                for message in messages {
                    if matches!(message, crate::gateway::websocket::WsMessage::Close { .. }) {
                        return true;
                    }
                }
            }
            Ok(Err(_)) => return true,
            // Nothing arrived, which is the normal case for a stream the
            // client only reads from
            Err(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_pressure::pressure::WorkloadClass;

    fn parse_pairs(json: &str) -> Vec<(String, String)> {
        // Enough of a reader to assert on a document this module wrote,
        // without taking a JSON parser dependency into the test
        let mut out = Vec::new();
        let bytes: Vec<char> = json.chars().collect();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == '"' {
                let start = i + 1;
                let mut end = start;
                while end < bytes.len() && bytes[end] != '"' {
                    end += 1;
                }
                let key: String = bytes[start..end].iter().collect();
                if end + 1 < bytes.len() && bytes[end + 1] == ':' {
                    let mut vend = end + 2;
                    let mut depth = 0;
                    while vend < bytes.len() {
                        match bytes[vend] {
                            '{' | '[' => depth += 1,
                            '}' | ']' if depth == 0 => break,
                            '}' | ']' => depth -= 1,
                            ',' if depth == 0 => break,
                            _ => {}
                        }
                        vend += 1;
                    }
                    let value: String = bytes[end + 2..vend].iter().collect();
                    out.push((key, value.trim().to_string()));
                }
                i = end + 1;
            } else {
                i += 1;
            }
        }
        out
    }

    #[test]
    fn the_snapshot_names_every_class() {
        let json = render_snapshot(None);
        for class in WorkloadClass::ALL {
            assert!(
                json.contains(&format!("\"class\":\"{}\"", class.as_str())),
                "{} is missing from the document",
                class.as_str()
            );
        }
    }

    /// JSON has no infinity, and Background's objective is genuinely
    /// unbounded, so it has to be null rather than a token no parser accepts.
    #[test]
    fn an_unbounded_objective_is_null_not_infinity() {
        let json = render_snapshot(None);
        assert!(!json.contains("inf"), "an infinity reached the document");
        assert!(!json.contains("NaN"));
        assert!(json.contains("\"slo_seconds\":null"));
        assert_eq!(number(f64::INFINITY), "null");
        assert_eq!(number(f64::NAN), "null");
    }

    #[test]
    fn the_document_is_balanced_and_carries_the_node_state() {
        let json = render_snapshot(None);
        let opens = json.chars().filter(|c| *c == '{').count();
        let closes = json.chars().filter(|c| *c == '}').count();
        assert_eq!(opens, closes, "unbalanced braces: {json}");
        let brackets_open = json.chars().filter(|c| *c == '[').count();
        let brackets_close = json.chars().filter(|c| *c == ']').count();
        assert_eq!(brackets_open, brackets_close);

        let pairs = parse_pairs(&json);
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        for expected in [
            "node_id",
            "sequence",
            "overall_utilization",
            "bottleneck",
            "actuator",
            "classes",
        ] {
            assert!(keys.contains(&expected), "{expected} missing from {keys:?}");
        }
    }

    #[test]
    fn hardware_appears_only_when_it_was_probed() {
        assert!(!render_snapshot(None).contains("\"hardware\""));
        let caps = NodeCapabilities::probe(1, None);
        let json = render_snapshot(Some(&caps));
        assert!(json.contains("\"hardware\""));
        assert!(json.contains(&caps.fingerprint.to_hex()));
    }

    /// A CPU model containing a quote would otherwise break the document.
    #[test]
    fn strings_are_escaped() {
        assert_eq!(escape("a\"b"), "a\\\"b");
        assert_eq!(escape("a\\b"), "a\\\\b");
        assert_eq!(escape("a\nb"), "a\\nb");
        let mut caps = NodeCapabilities::probe(1, None);
        caps.inputs.cpu_model = "Weird \"CPU\"\\model".to_string();
        let json = render_snapshot(Some(&caps));
        let opens = json.chars().filter(|c| *c == '{').count();
        let closes = json.chars().filter(|c| *c == '}').count();
        assert_eq!(opens, closes, "escaping broke the document: {json}");
    }

    /// The document is hand built, so it is parsed rather than eyeballed.
    ///
    /// Balanced braces are not valid JSON: a missing comma between two objects
    /// balances perfectly and parses nowhere. Every consumer of this endpoint
    /// is a parser, so the test is one too.
    #[test]
    fn the_document_parses_as_json() {
        let caps = NodeCapabilities::probe(9, None);
        for capabilities in [None, Some(&caps)] {
            let body = render_snapshot(capabilities);
            let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or_else(|e| {
                panic!(
                    "the pressure document is not JSON: {e}
{body}"
                )
            });

            let object = parsed.as_object().expect("a document");
            for required in [
                "node_id",
                "classes",
                "parallel",
                "query_memory_pct",
                "memory",
                "connections",
                "provisioner",
                "contention",
                "hot_set",
                "checkpoint",
            ] {
                assert!(object.contains_key(required), "no {required} in {body}");
            }

            // Every class carries its projection, which is what a scale
            // decision is taken from
            let classes = object["classes"].as_array().expect("classes");
            assert_eq!(classes.len(), WorkloadClass::COUNT);
            for class in classes {
                let projected = class
                    .get("projected")
                    .and_then(|p| p.as_object())
                    .unwrap_or_else(|| panic!("no projection on {class}"));
                assert!(projected.contains_key("horizon_seconds"));
                assert!(projected.contains_key("warm_pool_nodes"));
            }

            // The readiness answer appears exactly when the hardware it is
            // computed from does
            assert_eq!(
                object.contains_key("scale_to_zero"),
                capabilities.is_some(),
                "the readiness answer did not follow the hardware section"
            );
        }
    }

    #[test]
    fn only_our_paths_are_claimed() {
        assert!(is_pressure_path("/pressure"));
        assert!(is_pressure_path("/pressure/stream"));
        assert!(!is_pressure_path("/pressures"));
        assert!(!is_pressure_path("/metrics"));
        assert!(!is_pressure_path("/health/live"));
    }
}
