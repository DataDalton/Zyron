//! What laying fold output out by a measured key buys over sorting it by
//! the primary key.
//!
//! One fold cycle writes one file, and it writes the same rows either way,
//! so its segment bounds are identical under both orderings. What the
//! ordering changes is the zones inside it: a zone covers 1024 rows and
//! records their bounds, so grouping the values a query filters on makes
//! most zones unable to hold a match. The scan turns that into a skipped
//! segment, because a segment whose every zone rejects the predicate holds
//! no matching row.
//!
//! The file-level twin of this property lives in the lake's maintenance
//! tests, where a clustering pass rewrites many files at once and the unit
//! that stops being opened is a file.

use std::collections::BTreeSet;

use zyron_common::{ClusterKey, ClusterStrategy, TypeId};
use zyron_storage::columnar::{
    BloomPolicy, ColumnDescriptor, CompactionConfig, CompactionInput, ZONE_MAP_BATCH_SIZE,
    ZyrFileReader, run_compaction_cycle,
};

const ROWS: usize = 4_096;
const REGIONS: i64 = 4;
const REGION_STEP: i64 = 1_000;

const COL_ID: u32 = 0;
const COL_REGION: u32 = 1;

/// Two columns: an ascending primary key and a region that cycles through
/// four widely separated values, so consecutive primary keys land in
/// different regions
fn input(cluster_keys: Vec<ClusterKey>) -> CompactionInput {
    let ids: Vec<Option<Vec<u8>>> = (0..ROWS as i64)
        .map(|r| Some(r.to_le_bytes().to_vec()))
        .collect();
    let regions: Vec<Option<Vec<u8>>> = (0..ROWS as i64)
        .map(|r| Some(((r % REGIONS) * REGION_STEP).to_le_bytes().to_vec()))
        .collect();
    CompactionInput {
        columns: vec![
            ColumnDescriptor {
                column_id: COL_ID,
                type_id: TypeId::Int64,
                value_size: 8,
                is_primary_key: true,
                bloom_policy: BloomPolicy::Suppress,
            },
            ColumnDescriptor {
                column_id: COL_REGION,
                type_id: TypeId::Int64,
                value_size: 8,
                is_primary_key: false,
                bloom_policy: BloomPolicy::Suppress,
            },
        ],
        column_data: vec![ids, regions],
        table_id: 1,
        xmin_lo: 1,
        xmin_hi: 1,
        cluster_keys,
    }
}

/// The fold the server runs. Only the file cap moves, so the whole row
/// set lands in one file and the zone comparison below is between two
/// layouts of the same rows rather than two different file splits
fn config(dir: &std::path::Path) -> CompactionConfig {
    CompactionConfig {
        max_rows_per_file: ROWS as u64,
        ..zyron_bench_harness::compaction_config(dir)
    }
}

/// Reads a signed little endian value out of a 32-byte stat slot
fn slot_i64(slot: &[u8; 32]) -> i64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&slot[..8]);
    i64::from_le_bytes(buf)
}

/// Zones whose bounds could hold `value`, which is what the scan counts
/// before deciding whether to open a segment at all
fn zones_admitting(path: &std::path::Path, column_id: u32, value: i64) -> usize {
    let reader = ZyrFileReader::open(path).expect("open .zyr");
    let rows = reader.row_count() as usize;
    let (_, zones) = reader
        .read_segment_metadata(column_id, rows)
        .expect("segment metadata");
    zones
        .iter()
        .filter(|z| slot_i64(&z.min_value) <= value && value <= slot_i64(&z.max_value))
        .count()
}

fn segment_bounds(path: &std::path::Path, column_id: u32) -> (i64, i64) {
    let reader = ZyrFileReader::open(path).expect("open .zyr");
    let rows = reader.row_count() as usize;
    let (header, _) = reader
        .read_segment_metadata(column_id, rows)
        .expect("segment metadata");
    (slot_i64(&header.min_value), slot_i64(&header.max_value))
}

/// Every value of one column, in file order
fn column_values(path: &std::path::Path, column_id: u32) -> Vec<i64> {
    let reader = ZyrFileReader::open(path).expect("open .zyr");
    let rows = reader.row_count() as usize;
    let (bytes, _) = reader.decode_column(column_id, rows, 8).expect("decode");
    (0..rows)
        .map(|r| {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[r * 8..(r + 1) * 8]);
            i64::from_le_bytes(buf)
        })
        .collect()
}

#[test]
fn test_fold_with_adaptive_ordering_prunes_more_zones_than_single_pk_sort() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pk_dir = dir.path().join("pk");
    let clustered_dir = dir.path().join("clustered");
    std::fs::create_dir_all(&pk_dir).expect("pk dir");
    std::fs::create_dir_all(&clustered_dir).expect("clustered dir");

    // No declared key falls back to the ascending primary key, which is
    // the bootstrap policy a table with no measured proposal gets
    let by_pk = run_compaction_cycle(&config(&pk_dir), input(Vec::new())).expect("pk fold");
    let by_region = run_compaction_cycle(
        &config(&clustered_dir),
        input(vec![ClusterKey {
            column_id: COL_REGION,
            strategy: ClusterStrategy::RangePartition,
            param: 0,
        }]),
    )
    .expect("clustered fold");

    assert_eq!(by_pk.row_count, ROWS as u64);
    assert_eq!(by_region.row_count, ROWS as u64);
    let zones = ROWS.div_ceil(ZONE_MAP_BATCH_SIZE as usize);
    assert!(zones > 1, "the file has to hold more than one zone");

    // Same rows either way, so nothing about the file's own bounds moved.
    // Whatever the ordering buys, it does not come from here
    assert_eq!(
        segment_bounds(&by_pk.file_path, COL_REGION),
        segment_bounds(&by_region.file_path, COL_REGION),
        "the ordering does not change which values the file holds"
    );

    // A value inside the file's range that no row holds. Under the primary
    // key sort every zone spans the whole region range and admits it, so
    // the segment is opened and decoded to find nothing
    let absent = REGION_STEP + REGION_STEP / 2;
    let pk_admitting = zones_admitting(&by_pk.file_path, COL_REGION, absent);
    let clustered_admitting = zones_admitting(&by_region.file_path, COL_REGION, absent);
    assert_eq!(
        pk_admitting, zones,
        "an ascending primary key scatters the regions across every zone"
    );
    assert_eq!(
        clustered_admitting, 0,
        "grouping the regions leaves no zone that could hold {}",
        absent
    );
    assert!(
        clustered_admitting < pk_admitting,
        "the measured ordering has to reject strictly more"
    );

    // A value rows do hold reaches one zone rather than all of them
    let present = REGION_STEP;
    assert_eq!(
        zones_admitting(&by_pk.file_path, COL_REGION, present),
        zones,
        "every zone can hold a present value under the primary key sort"
    );
    let clustered_present = zones_admitting(&by_region.file_path, COL_REGION, present);
    assert!(
        clustered_present < zones,
        "grouping the regions confines a present value: {} zones of {}",
        clustered_present,
        zones
    );

    // The ordering moved rows and lost none of them
    let pk_regions = column_values(&by_pk.file_path, COL_REGION);
    let clustered_regions = column_values(&by_region.file_path, COL_REGION);
    assert_ne!(pk_regions, clustered_regions, "the layout actually changed");
    let mut pk_sorted = pk_regions.clone();
    let mut clustered_sorted = clustered_regions.clone();
    pk_sorted.sort_unstable();
    clustered_sorted.sort_unstable();
    assert_eq!(pk_sorted, clustered_sorted, "every row survived the fold");

    let pk_ids: BTreeSet<i64> = column_values(&by_pk.file_path, COL_ID)
        .into_iter()
        .collect();
    let clustered_ids: BTreeSet<i64> = column_values(&by_region.file_path, COL_ID)
        .into_iter()
        .collect();
    assert_eq!(pk_ids, clustered_ids, "the primary keys are the same set");
    assert_eq!(pk_ids.len(), ROWS);
}
