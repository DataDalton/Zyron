//! What one node learns, the fleet knows.
//!
//! Calibration is measured from real traffic, which means a node that has just
//! started has none. Persisting it to disk fixes the restart case and does
//! nothing for the case that matters more: a node added to a running mesh, on
//! hardware identical to the ten already serving, which should not have to
//! rediscover what a row costs on this instance type.
//!
//! So the measurement travels. It is keyed by hardware shape, so only nodes it
//! describes will adopt it, and by publishing node, so two nodes measuring the
//! same shape do not overwrite each other. Each reader merges what it finds,
//! weighted by how much traffic stands behind it.
//!
//! Run: cargo test -p zyron-server --test calibration_gossip_test

use std::collections::HashMap;
use std::sync::Arc;

use zyron_pressure::capability::{
    CalibrationField, HardwareFingerprint, NodeCapabilities, OperatorCoefficients, OperatorKind,
    calibration_key, decode_coefficient, encode_coefficient, parse_calibration_key,
};
use zyron_pressure::pressure::{clamp_payload, decode_versioned, encode_versioned};
use zyron_pressure::pressure_control::PressureController;
use zyron_server::background::pressure::{adopt_peer_calibration, publish_calibration};
use zyron_types::scheduling::QuotaRegistry;

/// The shape of machine the fleet is running on.
const FLEET: HardwareFingerprint = HardwareFingerprint(0xABCD_1234_5678_9F01);

/// A different shape, whose measurements must not be adopted.
const OTHER_HARDWARE: HardwareFingerprint = HardwareFingerprint(0x0FED_CBA9_8765_4321);

/// The kind the fleet is learning about.
const KIND: OperatorKind = OperatorKind::Sort;

fn capabilities(fingerprint: HardwareFingerprint) -> NodeCapabilities {
    let mut caps = NodeCapabilities::probe(0, None);
    caps.fingerprint = fingerprint;
    caps
}

/// One node in the fleet, with its own controller and its own view of the
/// shared registry.
struct Node {
    controller: PressureController,
    capabilities: NodeCapabilities,
    folded: HashMap<(u64, usize), u64>,
}

impl Node {
    fn new(node_id: u64, fingerprint: HardwareFingerprint) -> Self {
        Self {
            controller: PressureController::new(node_id, 8 * 1024 * 1024 * 1024),
            capabilities: capabilities(fingerprint),
            folded: HashMap::new(),
        }
    }

    /// Records real work, which is the only thing that produces a coefficient.
    fn measure(&self, kind: OperatorKind, batches: u64, nanos_per_unit: u64) {
        self.controller.coefficients().record(
            kind,
            batches,
            batches.saturating_mul(nanos_per_unit),
        );
        self.controller.drain_calibration();
    }

    fn publish(&self, registry: &QuotaRegistry) -> usize {
        publish_calibration(&self.controller, registry, &self.capabilities)
    }

    fn adopt(&mut self, registry: &QuotaRegistry) -> usize {
        adopt_peer_calibration(
            &self.controller,
            registry,
            &self.capabilities,
            &mut self.folded,
        )
    }

    /// What the node prices a plan against: its own measurement merged with
    /// the fleet's.
    fn believes(&self, kind: OperatorKind) -> f64 {
        self.controller.coefficients().effective().get(kind)
    }

    /// How much evidence stands behind that belief, from both sources.
    fn evidence(&self, kind: OperatorKind) -> u64 {
        self.controller.coefficients().effective().samples(kind)
    }

    /// What the node measured itself, which is the only thing it publishes.
    fn measured(&self, kind: OperatorKind) -> u64 {
        self.controller.coefficients().samples(kind)
    }
}

/// One gossip round: everyone publishes, then everyone reads.
fn round(nodes: &mut [Node], registry: &QuotaRegistry) {
    for node in nodes.iter() {
        node.publish(registry);
    }
    for node in nodes.iter_mut() {
        node.adopt(registry);
    }
}

/// Three nodes, one of which has measured something. Within two rounds the
/// other two are pricing plans against it.
#[test]
fn a_measurement_reaches_the_fleet_within_two_rounds() {
    let registry = Arc::new(QuotaRegistry::new());
    let mut nodes = vec![
        Node::new(1, FLEET),
        Node::new(2, FLEET),
        Node::new(3, FLEET),
    ];

    // Only the first node has run any sorts, and they cost 40ns a row here
    nodes[0].measure(KIND, 5_000, 40);
    let measured = nodes[0].believes(KIND);
    assert!(
        (measured - 40.0).abs() < 1.0,
        "the measuring node itself believes {measured}"
    );

    let cold = nodes[1].believes(KIND);
    assert!(
        (cold - measured).abs() > 1.0,
        "the other nodes already agreed before any gossip, so this proves nothing"
    );

    round(&mut nodes, &registry);
    round(&mut nodes, &registry);

    for node in &nodes[1..] {
        let adopted = node.believes(KIND);
        assert!(
            (adopted - measured).abs() / measured < 0.05,
            "a peer still believes {adopted} where the fleet measured {measured}"
        );
        assert!(
            node.evidence(KIND) >= 5_000,
            "the peer took the value without the evidence behind it"
        );
    }
}

/// Evidence is counted once. A peer read ten times must not end up outweighing
/// the rest of the fleet ten to one.
#[test]
fn a_peer_read_repeatedly_is_not_counted_repeatedly() {
    let registry = Arc::new(QuotaRegistry::new());
    let mut nodes = vec![Node::new(1, FLEET), Node::new(2, FLEET)];
    nodes[0].measure(KIND, 1_000, 60);

    round(&mut nodes, &registry);
    let after_first = nodes[1].evidence(KIND);
    assert!(after_first >= 1_000);

    for _ in 0..10 {
        round(&mut nodes, &registry);
    }
    let after_ten_more = nodes[1].evidence(KIND);
    assert!(
        after_ten_more < after_first * 2,
        "one peer's evidence grew from {after_first} to {after_ten_more} by being read"
    );
}

/// New evidence from a peer does move the number, so the fleet keeps tracking
/// a machine whose behaviour changes.
#[test]
fn fresh_peer_evidence_still_moves_the_answer() {
    let registry = Arc::new(QuotaRegistry::new());
    let mut nodes = vec![Node::new(1, FLEET), Node::new(2, FLEET)];

    nodes[0].measure(KIND, 2_000, 20);
    round(&mut nodes, &registry);
    let before = nodes[1].believes(KIND);

    // The volume the fleet runs on slows down, and the node still serving
    // sorts is the one that sees it first
    for _ in 0..20 {
        nodes[0].measure(KIND, 5_000, 200);
    }
    for _ in 0..3 {
        round(&mut nodes, &registry);
    }
    let after = nodes[1].believes(KIND);
    assert!(
        after > before * 2.0,
        "the peer did not follow a fivefold slowdown: {before} to {after}"
    );
}

/// A measurement from different hardware is not adopted, whatever it says.
#[test]
fn a_different_machine_shape_is_ignored() {
    let registry = Arc::new(QuotaRegistry::new());
    let mut fleet_node = Node::new(1, FLEET);
    let stranger = Node::new(2, OTHER_HARDWARE);

    stranger.measure(KIND, 100_000, 9_000);
    stranger.publish(&registry);
    let before = fleet_node.believes(KIND);

    let adopted = fleet_node.adopt(&registry);
    assert_eq!(
        adopted, 0,
        "a node adopted a measurement taken on hardware it is not running on"
    );
    assert_eq!(fleet_node.believes(KIND), before);
}

/// A node does not adopt its own record back, which would double its own
/// evidence every round.
#[test]
fn a_node_does_not_read_back_its_own_measurement() {
    let registry = Arc::new(QuotaRegistry::new());
    let mut node = Node::new(1, FLEET);
    node.measure(KIND, 3_000, 55);
    let evidence_before = node.evidence(KIND);

    node.publish(&registry);
    assert_eq!(node.adopt(&registry), 0);
    assert_eq!(node.evidence(KIND), evidence_before);
    assert_eq!(node.measured(KIND), 3_000);
}

/// A kind nothing has measured is not published, so a cold-start constant does
/// not spread through the fleet dressed as a reading.
#[test]
fn unmeasured_kinds_are_not_published() {
    let registry = Arc::new(QuotaRegistry::new());
    let node = Node::new(1, FLEET);
    assert_eq!(node.publish(&registry), 0);

    node.measure(KIND, 500, 30);
    // One kind, two fields
    assert_eq!(node.publish(&registry), 2);

    let published: Vec<_> = registry
        .snapshot()
        .into_iter()
        .filter_map(|(k, _)| parse_calibration_key(&k))
        .collect();
    assert_eq!(published.len(), 2);
    assert!(published.iter().all(|(_, _, kind, _)| *kind == KIND));
}

/// The key format survives a round trip, including every field and kind.
#[test]
fn calibration_keys_round_trip() {
    for kind in OperatorKind::ALL {
        for field in CalibrationField::ALL {
            let key = calibration_key(FLEET, 77, kind, field);
            assert_eq!(
                parse_calibration_key(&key),
                Some((FLEET, 77, kind, field)),
                "{key}"
            );
        }
    }
    assert_eq!(parse_calibration_key("zpressure:1:interactive:p"), None);
    assert_eq!(parse_calibration_key("zcalib:notahexvalue:1:0:v"), None);
    assert_eq!(parse_calibration_key("zcalib:1:2:99:v"), None);
}

/// The payload keeps enough resolution for a sub-nanosecond filter and enough
/// range for a slow fsync, which are the two ends of what is measured.
#[test]
fn the_payload_holds_both_ends_of_the_range() {
    for ns in [0.05f64, 1.0, 40.0, 80_000.0, 40_000_000.0] {
        let packed = encode_coefficient(ns);
        let back = decode_coefficient(packed);
        assert!((back - ns).abs() / ns < 0.01, "{ns} came back as {back}");
    }
    assert_eq!(encode_coefficient(-1.0), 0);
    assert_eq!(encode_coefficient(f64::NAN), 0);
}

/// The record carries a sequence, so a stale reading published earlier cannot
/// displace a newer one under the registry's maximum merge.
#[test]
fn a_newer_record_wins_the_merge() {
    let registry = QuotaRegistry::new();
    let key = calibration_key(FLEET, 4, KIND, CalibrationField::CentiNanosPerUnit);
    registry.merge_remote(&[(key.clone(), encode_versioned(9, encode_coefficient(500.0)))]);
    // A later tick reporting a much lower cost, which a plain maximum would
    // discard for being smaller
    registry.merge_remote(&[(key.clone(), encode_versioned(10, encode_coefficient(3.0)))]);

    let stored = registry
        .snapshot()
        .into_iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
        .expect("the record");
    let (sequence, payload) = decode_versioned(stored);
    assert_eq!(sequence, 10);
    assert!((decode_coefficient(payload) - 3.0).abs() < 0.1);
}

/// Two nodes with different amounts of traffic behind them produce a weighted
/// answer, not whichever was read last.
#[test]
fn the_merge_is_weighted_by_evidence() {
    let mut heavy = OperatorCoefficients::cold_start();
    heavy.ns_per_unit[KIND.index()] = 10.0;
    heavy.sample_count[KIND.index()] = 90_000;

    let mut light = OperatorCoefficients::cold_start();
    light.ns_per_unit[KIND.index()] = 110.0;
    light.sample_count[KIND.index()] = 10_000;

    heavy.merge(&light);
    // Nine tenths of the evidence says ten, so the answer sits near twenty
    assert!(
        (heavy.get(KIND) - 20.0).abs() < 0.001,
        "{}",
        heavy.get(KIND)
    );
    assert_eq!(heavy.samples(KIND), 100_000);
}

/// The sample count is carried through the payload without losing the
/// magnitude that decides the weighting.
#[test]
fn the_sample_count_survives_the_payload() {
    for samples in [1u64, 1_000, 1_000_000, u32::MAX as u64] {
        assert_eq!(clamp_payload(samples) as u64, samples);
    }
    // Past the payload width it saturates rather than wrapping to something
    // small, which would make a heavily measured node look inexperienced
    assert_eq!(clamp_payload(u64::MAX), u32::MAX);
}
