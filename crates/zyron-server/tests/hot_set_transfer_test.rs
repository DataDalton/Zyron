//! The working set crossing from a draining node to a survivor.
//!
//! The manifest itself is tested where it is built, and the prefetch is tested
//! where the buffer pool lives. What is only testable from here is that the
//! two halves agree across the wire: what the departing node serves is what
//! the survivor can read, and a survivor that fetches it ends up holding the
//! pages the departing node was holding.
//!
//! Run: cargo test -p zyron-server --test hot_set_transfer_test

use zyron_common::page::PageId;
use zyron_pressure::hot_set::{HotQuery, HotSetManifest};
use zyron_server::gateway::pressure_endpoint::{parse_hot_set, render_hot_set};

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "zyron_hot_set_transfer_{}_{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn manifest(node_id: u64, pages: u64) -> HotSetManifest {
    HotSetManifest::build(
        node_id,
        1_700_000_000_000_000,
        (0..pages).map(|n| PageId::new((n % 4) as u32, n)),
        u32::MAX,
        vec![
            HotQuery {
                fingerprint: 0x1234_5678_9ABC_DEF0,
                executions: 900,
                mean_work_us: 4_000,
            },
            HotQuery {
                fingerprint: 0x0FED_CBA9_8765_4321,
                executions: 12,
                mean_work_us: 900_000,
            },
        ],
        16,
    )
}

/// What the departing node serves is exactly what the survivor reads back.
#[test]
fn a_served_manifest_round_trips_to_a_survivor() {
    let dir = scratch("round_trip");
    let original = manifest(11, 20_000);
    original.persist(&dir).expect("persist");

    let (status, body) = render_hot_set(&dir);
    assert_eq!(status, "200 OK", "{body}");
    assert!(body.contains("\"node_id\":11"), "{body}");
    assert!(body.contains("\"pages\":20000"), "{body}");

    let received = parse_hot_set(&body).expect("the survivor could not read the manifest");
    assert_eq!(received, original);
    assert_eq!(received.queries.len(), 2);
    // Ranked by total work, so the twelve expensive runs come first
    assert_eq!(received.queries[0].fingerprint, 0x0FED_CBA9_8765_4321);

    let _ = std::fs::remove_dir_all(&dir);
}

/// A node with no manifest says so rather than serving an empty one.
///
/// The two answers lead a survivor to opposite actions. An empty manifest
/// means the departing node was holding nothing and the traffic can move now.
/// A missing one means the handover is not ready, and moving traffic would
/// move it to a cold pool.
#[test]
fn a_node_without_a_manifest_reports_that_rather_than_an_empty_one() {
    let dir = scratch("absent");
    let (status, body) = render_hot_set(&dir);
    assert_eq!(status, "404 Not Found");
    assert!(body.contains("has not written"), "{body}");
    assert_eq!(parse_hot_set(&body), None);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A manifest that cannot be trusted is refused rather than partly served.
#[test]
fn a_corrupted_manifest_is_reported_as_an_error() {
    let dir = scratch("corrupt");
    manifest(3, 500).persist(&dir).expect("persist");
    let path = zyron_pressure::hot_set::manifest_path(&dir);
    let mut bytes = std::fs::read(&path).expect("read");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("write");

    let (status, body) = render_hot_set(&dir);
    assert_eq!(status, "500 Internal Server Error", "{body}");
    assert!(body.contains("could not be read"), "{body}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A large working set still fits a response a survivor can fetch in one go,
/// which is what makes the handover practical rather than theoretical.
#[test]
fn a_large_working_set_transfers_as_a_small_response() {
    let dir = scratch("large");
    // Four gigabytes of sixteen kilobyte pages
    let big = manifest(5, 262_144);
    big.persist(&dir).expect("persist");

    let (status, body) = render_hot_set(&dir);
    assert_eq!(status, "200 OK");
    // Under a megabyte for a working set that describes gigabytes, because
    // the identifiers are sorted and delta encoded before they are base64
    assert!(
        body.len() < 1_000_000,
        "a 262144 page manifest served {} bytes",
        body.len()
    );
    assert_eq!(parse_hot_set(&body).expect("decode").pages.len(), 262_144);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The route is one this module owns, so the health listener answers it rather
/// than falling through to the not-found path.
#[test]
fn the_route_is_recognised() {
    use zyron_server::gateway::pressure_endpoint::{HOT_SET_PATH, is_pressure_path};
    assert!(is_pressure_path(HOT_SET_PATH));
    assert!(!is_pressure_path("/pressure/nothing"));
}

/// Scale to zero is refused until both preconditions actually hold, and the
/// two states are read from the node rather than declared.
#[test]
fn scale_to_zero_reads_the_node_rather_than_being_told() {
    use zyron_pressure::capability::NodeCapabilities;
    use zyron_pressure::pressure_control::PressureController;
    use zyron_server::background::pressure::{resume_estimate, scale_to_zero_readiness};

    let controller = PressureController::new(1, 8 * 1024 * 1024 * 1024);
    let capabilities = NodeCapabilities::probe(1, None);

    // Nothing checkpointed and nothing persisted, so both refuse
    assert!(scale_to_zero_readiness(&controller, &capabilities).is_err());

    // A checkpoint that never ran is not a clean one, whatever was recorded
    controller.record_checkpoint(0, true);
    assert!(
        scale_to_zero_readiness(&controller, &capabilities).is_err(),
        "a node that has never checkpointed was allowed to stop"
    );

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    controller.record_checkpoint(now_us, true);
    controller.record_wal_since_checkpoint(4_000_000);
    let refused = scale_to_zero_readiness(&controller, &capabilities)
        .expect_err("no manifest, so it must still refuse");
    assert!(refused.to_string().contains("manifest"), "{refused}");

    controller.record_hot_set(20_000, now_us, true);
    let estimate =
        scale_to_zero_readiness(&controller, &capabilities).expect("both preconditions hold");
    assert!(estimate.total >= estimate.checkpoint_freshness);
    assert_eq!(
        estimate,
        resume_estimate(&controller, &capabilities),
        "the estimate changed depending on whether it was allowed"
    );

    // A checkpoint that failed takes the permission away again
    controller.record_checkpoint(now_us, false);
    let after_failure = scale_to_zero_readiness(&controller, &capabilities)
        .expect_err("a failed checkpoint must withdraw permission");
    assert!(
        after_failure.to_string().contains("checkpoint"),
        "{after_failure}"
    );
}

/// A clean checkpoint resets the log divergence, because the log written
/// before it is exactly what the checkpoint made unnecessary.
#[test]
fn a_clean_checkpoint_resets_the_log_divergence() {
    use zyron_pressure::pressure_control::PressureController;

    let controller = PressureController::new(1, 8 * 1024 * 1024 * 1024);
    controller.record_wal_since_checkpoint(9_000_000);
    assert_eq!(controller.checkpoint_state().wal_bytes_since, 9_000_000);

    controller.record_checkpoint(1_700_000_000_000_000, true);
    assert_eq!(controller.checkpoint_state().wal_bytes_since, 0);

    // A failed one does not, because nothing was made unnecessary
    controller.record_wal_since_checkpoint(3_000_000);
    controller.record_checkpoint(1_700_000_000_100_000, false);
    assert_eq!(controller.checkpoint_state().wal_bytes_since, 3_000_000);
    assert!(!controller.checkpoint_state().clean);
}
