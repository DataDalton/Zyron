//! Round-trip codec tests for the data lifecycle catalog entries.

use zyron_catalog::schema::{
    ColumnarRegistry, ColumnarSegmentEntry, ComplianceLogEntry, LakeConfig, LegalHoldEntry,
    LifecycleConfig, RetentionJobEntry, RetentionPolicyEntry,
};
use zyron_catalog::{SchemaId, TableEntry, TableId};

#[test]
fn table_entry_lifecycle_roundtrip() {
    let mut lc = LifecycleConfig::default();
    lc.soft_delete_enabled = true;
    lc.soft_delete_is_deleted_col_id = 3;
    lc.soft_delete_deleted_at_col_id = 4;
    lc.ttl_column_id = 5;
    lc.ttl_seconds = 90 * 86400;
    lc.ttl_action = 2;
    lc.retention_column_id = 6;
    lc.storage_tier = 2;
    lc.cold_after_seconds = 30 * 86400;
    lc.archive_after_seconds = 365 * 86400;
    lc.archive_destination = "s3://bucket/arch/".into();
    lc.archive_on_purge = true;
    lc.purge_grace_seconds = 7 * 86400;
    lc.retention_lock_until = 1_700_000_000_000_000;
    lc.recycle_window_seconds = 3600;
    lc.data_key_id = 42;
    lc.residency_region = "EU".into();
    lc.immutable = true;

    let entry = TableEntry {
        id: TableId(200),
        schema_id: SchemaId(1),
        name: "events".into(),
        heap_file_id: 200,
        fsm_file_id: 201,
        columns: vec![],
        constraints: vec![],
        created_at: 12345,
        versioning_enabled: false,
        scd_type: None,
        system_versioned: false,
        history_table_id: None,
        cdf_enabled: false,
        cdf_retention_days: 0,
        lifecycle: lc.clone(),
        columnar: Default::default(),
        dropped_at: None,
        expectations: Vec::new(),
        time_travel_retention_secs: 0,
        lake: Default::default(),
        cluster: Default::default(),
        foreign: Default::default(),
    };
    let bytes = entry.to_bytes();
    let decoded = TableEntry::from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.lifecycle, lc);
    assert_eq!(decoded.name, "events");
}

#[test]
fn table_entry_backward_compatible_without_lifecycle() {
    // A byte image with no lifecycle section: re-encode an entry whose
    // lifecycle is default and confirm a short buffer still decodes with
    // defaults.
    let entry = TableEntry {
        id: TableId(1),
        schema_id: SchemaId(1),
        name: "old".into(),
        heap_file_id: 200,
        fsm_file_id: 201,
        columns: vec![],
        constraints: vec![],
        created_at: 1,
        versioning_enabled: false,
        scd_type: None,
        system_versioned: false,
        history_table_id: None,
        cdf_enabled: false,
        cdf_retention_days: 0,
        lifecycle: LifecycleConfig::default(),
        columnar: Default::default(),
        dropped_at: None,
        expectations: Vec::new(),
        time_travel_retention_secs: 0,
        lake: Default::default(),
        cluster: Default::default(),
        foreign: Default::default(),
    };
    let decoded = TableEntry::from_bytes(&entry.to_bytes()).expect("decode");
    assert_eq!(decoded.lifecycle, LifecycleConfig::default());
}

#[test]
fn table_entry_columnar_registry_roundtrip() {
    let columnar = ColumnarRegistry {
        segments: vec![
            ColumnarSegmentEntry {
                file_id: 1,
                path: "data/columnar/t7_f1.zyr".into(),
                row_count: 1_048_576,
                sys_rowid_lo: 0,
                sys_rowid_hi: 1_048_575,
                sys_xmin_lo: 100,
                sys_xmin_hi: 9_500,
                cluster_spec_id: 1,
                storage_tier: 0,
            },
            ColumnarSegmentEntry {
                file_id: 2,
                path: "data/columnar/t7_f2.zyr".into(),
                row_count: 500_000,
                sys_rowid_lo: 1_048_576,
                sys_rowid_hi: 1_548_575,
                sys_xmin_lo: 9_600,
                sys_xmin_hi: 12_000,
                cluster_spec_id: 1,
                // A relocated segment has to survive the round trip too, or
                // a restart would read every cold file back as hot
                storage_tier: 2,
            },
        ],
        next_rowid: 1_548_576,
        next_file_id: 3,
        low_water: 11_800,
    };
    let entry = TableEntry {
        id: TableId(7),
        schema_id: SchemaId(1),
        name: "metrics".into(),
        heap_file_id: 200,
        fsm_file_id: 201,
        columns: vec![],
        constraints: vec![],
        created_at: 1,
        versioning_enabled: false,
        scd_type: None,
        system_versioned: false,
        history_table_id: None,
        cdf_enabled: false,
        cdf_retention_days: 0,
        lifecycle: LifecycleConfig::default(),
        columnar: columnar.clone(),
        dropped_at: None,
        expectations: Vec::new(),
        time_travel_retention_secs: 0,
        lake: Default::default(),
        cluster: Default::default(),
        foreign: Default::default(),
    };
    let decoded = TableEntry::from_bytes(&entry.to_bytes()).expect("decode");
    assert_eq!(decoded.columnar, columnar);
    // A buffer without the columnar tail decodes to an empty registry.
    assert_eq!(
        ColumnarRegistry::default(),
        ColumnarRegistry {
            segments: vec![],
            next_rowid: 0,
            next_file_id: 0,
            low_water: 0,
        }
    );
}

#[test]
fn table_entry_lake_tail_roundtrip() {
    let mut entry = TableEntry {
        id: TableId(9),
        schema_id: SchemaId(1),
        name: "lake_t".into(),
        heap_file_id: 0,
        fsm_file_id: 0,
        columns: vec![],
        constraints: vec![],
        created_at: 1,
        versioning_enabled: false,
        scd_type: None,
        system_versioned: false,
        history_table_id: None,
        cdf_enabled: false,
        cdf_retention_days: 0,
        lifecycle: LifecycleConfig::default(),
        columnar: ColumnarRegistry::default(),
        dropped_at: None,
        expectations: Vec::new(),
        time_travel_retention_secs: 0,
        lake: LakeConfig::lake(),
        cluster: Default::default(),
        foreign: Default::default(),
    };
    let decoded = TableEntry::from_bytes(&entry.to_bytes()).expect("decode");
    assert!(decoded.lake.is_lake());

    entry.lake = LakeConfig::default();
    let heap = TableEntry::from_bytes(&entry.to_bytes()).expect("decode heap");
    assert!(!heap.lake.is_lake());

    // Bytes written before a tail section existed decode to that section's
    // default. Each cut is measured rather than counted by hand, so adding a
    // later tail section does not silently move the earlier cuts and turn
    // this into a test of nothing
    let full = entry.to_bytes();
    // foreign: two u32-prefixed strings, both empty on a local table
    let foreign_len = 4 + 4;
    // cluster: mode, schedule, spec id, key count, no keys
    let cluster_len = 1 + 1 + 4 + 2;
    // lake: format, retained-history flag, then the leader it follows as two
    // u32-prefixed strings, both empty on a table that follows nobody
    let lake_len = 1 + 1 + 4 + 4;

    let pre_foreign = TableEntry::from_bytes(&full[..full.len() - foreign_len])
        .expect("decode pre-foreign bytes");
    assert!(
        !pre_foreign.foreign.is_foreign(),
        "a table written before federation existed is local, which is what it meant"
    );
    assert_eq!(pre_foreign.cluster.spec_id, entry.cluster.spec_id);

    let pre_cluster = TableEntry::from_bytes(&full[..full.len() - foreign_len - cluster_len])
        .expect("decode pre-clustering bytes");
    assert!(pre_cluster.cluster.keys.is_empty());
    assert_eq!(pre_cluster.cluster.spec_id, 0);
    assert!(!pre_cluster.foreign.is_foreign());

    let pre_lake =
        TableEntry::from_bytes(&full[..full.len() - foreign_len - cluster_len - lake_len])
            .expect("decode pre-lake bytes");
    assert!(!pre_lake.lake.is_lake());
}

/// The clustering policy is what the fold tier reads to lay out a heap
/// table's segments, so it has to survive a catalog round trip exactly
#[test]
fn table_entry_cluster_tail_roundtrip() {
    let mut entry = TableEntry {
        id: TableId(11),
        schema_id: SchemaId(1),
        name: "events".into(),
        heap_file_id: 0,
        fsm_file_id: 0,
        columns: vec![],
        constraints: vec![],
        created_at: 1,
        versioning_enabled: false,
        scd_type: None,
        system_versioned: false,
        history_table_id: None,
        cdf_enabled: false,
        cdf_retention_days: 0,
        lifecycle: LifecycleConfig::default(),
        columnar: ColumnarRegistry::default(),
        dropped_at: None,
        expectations: Vec::new(),
        time_travel_retention_secs: 0,
        lake: LakeConfig::default(),
        cluster: Default::default(),
        foreign: Default::default(),
    };
    entry.cluster.mode = zyron_common::ClusterMode::Hybrid.to_u8();
    entry.cluster.schedule = zyron_common::ClusteringSchedule::Continuous.to_u8();
    entry.cluster.set_keys(&[
        zyron_common::ClusterKey {
            column_id: 3,
            strategy: zyron_common::ClusterStrategy::BitInterleave,
            param: 0,
        },
        zyron_common::ClusterKey {
            column_id: 7,
            strategy: zyron_common::ClusterStrategy::SpaceFilling,
            param: 5,
        },
    ]);

    let decoded = TableEntry::from_bytes(&entry.to_bytes()).expect("decode");
    assert_eq!(decoded.cluster, entry.cluster);
    assert_eq!(decoded.cluster.mode(), zyron_common::ClusterMode::Hybrid);
    assert_eq!(
        decoded.cluster.schedule(),
        zyron_common::ClusteringSchedule::Continuous
    );
    let keys = decoded.cluster.fold_keys();
    assert_eq!(keys.len(), 2);
    assert_eq!(
        keys[0].strategy,
        zyron_common::ClusterStrategy::BitInterleave
    );
    assert_eq!(
        keys[1].strategy,
        zyron_common::ClusterStrategy::SpaceFilling
    );
    assert_eq!(keys[1].param, 5);

    // Declaring keys advances the spec, so a segment written before the
    // change is distinguishable from one written after it
    let before = decoded.cluster.spec_id;
    let mut again = decoded.clone();
    again.cluster.set_keys(&[]);
    assert!(again.cluster.spec_id > before);
    assert!(again.cluster.fold_keys().is_empty());

    // An unknown strategy byte costs ordering quality, not the fold
    let mut corrupt = entry.clone();
    corrupt.cluster.keys[0].strategy = 200;
    assert_eq!(
        corrupt.cluster.fold_keys()[0].strategy,
        zyron_common::ClusterStrategy::RangePartition
    );
}

#[test]
fn lifecycle_system_entries_roundtrip() {
    let h = LegalHoldEntry {
        id: 9,
        name: "case_1".into(),
        table_id: 7,
        predicate_sql: "(customer_id = 5)".into(),
        reason: "litigation".into(),
        created_at: 111,
        released_at: 0,
    };
    assert_eq!(LegalHoldEntry::from_bytes(&h.to_bytes()).unwrap(), h);
    assert!(h.is_active());

    let p = RetentionPolicyEntry {
        table_id: 7,
        kind: 0,
        interval_seconds: 7776000,
        action: 1,
        destination: "s3://x/".into(),
    };
    assert_eq!(RetentionPolicyEntry::from_bytes(&p.to_bytes()).unwrap(), p);

    let j = RetentionJobEntry {
        job_id: 1,
        table_id: 7,
        kind: 3,
        scheduled_at: 1,
        started_at: 2,
        finished_at: 3,
        rows_affected: 1234,
        status: 2,
        detail: "purge".into(),
    };
    assert_eq!(RetentionJobEntry::from_bytes(&j.to_bytes()).unwrap(), j);
}

#[test]
fn compliance_log_hash_chain() {
    let mut e1 = ComplianceLogEntry {
        event_id: 1,
        event_type: 3,
        subject: "h1".into(),
        table_id: 7,
        ts: 100,
        detail: "create legal hold".into(),
        prev_hash: 0,
        entry_hash: 0,
    };
    e1.entry_hash = e1.compute_hash();
    let mut e2 = ComplianceLogEntry {
        event_id: 2,
        event_type: 4,
        subject: "u-1".into(),
        table_id: 0,
        ts: 200,
        detail: "forget user".into(),
        prev_hash: e1.entry_hash,
        entry_hash: 0,
    };
    e2.entry_hash = e2.compute_hash();

    // Chain verifies.
    let mut prev = 0u32;
    for e in [&e1, &e2] {
        assert_eq!(e.prev_hash, prev);
        assert_eq!(e.entry_hash, e.compute_hash());
        prev = e.entry_hash;
    }
    // Round-trips through bytes.
    assert_eq!(ComplianceLogEntry::from_bytes(&e2.to_bytes()).unwrap(), e2);
    // Tampering breaks the recomputed hash.
    let mut tampered = e2.clone();
    tampered.detail = "forget user (changed)".into();
    assert_ne!(tampered.compute_hash(), tampered.entry_hash);
}

#[test]
fn constraint_entry_enforced_tail_roundtrip() {
    use zyron_catalog::schema::{ConstraintEntry, ConstraintType, ReferentialAction};
    use zyron_catalog::{ColumnId, TableId};

    let entry = ConstraintEntry {
        name: "uq_a".into(),
        constraint_type: ConstraintType::Unique,
        columns: vec![ColumnId(1)],
        ref_table_id: None,
        ref_columns: vec![],
        check_expr: None,
        on_delete: ReferentialAction::NoAction,
        on_update: ReferentialAction::NoAction,
        enforced: false,
        on_violation: zyron_catalog::schema::ConstraintViolationAction::Quarantine,
        quarantine_table_id: Some(77),
    };
    let bytes = entry.to_bytes();
    let mut off = 0usize;
    let decoded = ConstraintEntry::from_bytes(&bytes, &mut off).expect("decode");
    assert!(!decoded.enforced, "the mode survives the round trip");
    assert_eq!(
        decoded.on_violation,
        zyron_catalog::schema::ConstraintViolationAction::Quarantine
    );
    assert_eq!(decoded.quarantine_table_id, Some(77));
    assert_eq!(off, bytes.len(), "the tail byte is consumed");

    // Bytes written before the modes existed decode as enforced and Fail,
    // which is what they meant. The tail here is enforced, on_violation, the
    // quarantine presence byte and the table id
    let truncated = &bytes[..bytes.len() - 7];
    let mut off = 0usize;
    let decoded = ConstraintEntry::from_bytes(truncated, &mut off).expect("decode");
    assert!(decoded.enforced);
    assert_eq!(
        decoded.on_violation,
        zyron_catalog::schema::ConstraintViolationAction::Fail
    );
    assert!(decoded.quarantine_table_id.is_none());
    let _ = TableId(1);
}
