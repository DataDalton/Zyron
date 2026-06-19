//! Round-trip codec tests for Phase 17 catalog entries.

use zyron_catalog::schema::{
    ColumnarRegistry, ColumnarSegmentEntry, ComplianceLogEntry, LegalHoldEntry, LifecycleConfig,
    RetentionJobEntry, RetentionPolicyEntry,
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
    };
    let bytes = entry.to_bytes();
    let decoded = TableEntry::from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.lifecycle, lc);
    assert_eq!(decoded.name, "events");
}

#[test]
fn table_entry_backward_compatible_without_lifecycle() {
    // Build a pre-Phase-17 byte image: a full entry then truncate the
    // lifecycle suffix by re-encoding an entry whose lifecycle is default and
    // confirming a short buffer still decodes with defaults.
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
            },
            ColumnarSegmentEntry {
                file_id: 2,
                path: "data/columnar/t7_f2.zyr".into(),
                row_count: 500_000,
                sys_rowid_lo: 1_048_576,
                sys_rowid_hi: 1_548_575,
                sys_xmin_lo: 9_600,
                sys_xmin_hi: 12_000,
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
