//! Which access path answers a predicate on an indexed lake column.
//!
//! An index is a way to read less, so one that reads more than the scan it
//! replaces has to be declined. That decision is made from the manifest
//! with no IO, and it is only as good as the byte counts it compares. It
//! used to charge the scan a file's size divided by its column count,
//! which bills a narrow key column for its widest neighbour, so the wider
//! the rest of the row the more likely an index was taken on a column
//! whose bounds had already picked one file.
//!
//! Both sides now come from what each plan reads. These tests pin the two
//! directions and the fact that a plan says which one it took

mod common;

use std::sync::Arc;

use common::{
    analyze_lake_scan, create_test_server, exec_ddl, exec_dml, new_session, query_values,
    render_analyzed,
};
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

/// Rows per INSERT, and so per data file. Enough that the text column
/// outgrows the page a fixed width column of the same rows fits in, which
/// is the spread the cost model has to see
const ROWS_PER_FILE: i64 = 3000;
/// Data files the table holds
const FILES: i64 = 4;
/// Distinct values of the shuffled column, one per row of every file, so
/// every file's bounds span the whole domain and no bound rejects any file
const TAGS: i64 = ROWS_PER_FILE;

/// A lake table of `FILES` files whose `id` ascends with insertion and
/// whose `tag` repeats the same range in every file.
///
/// The two columns are the two cases. Bounds on `id` reach one file and
/// bounds on `tag` reach all of them, so an index on `id` has nothing left
/// to remove and an index on `tag` is the only thing that can remove
/// anything
async fn table_with_a_clustered_and_a_shuffled_column(
    server: &Arc<ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    name: &str,
) {
    exec_ddl(
        server,
        session,
        &format!(
            "CREATE TABLE {name} (id BIGINT NOT NULL, tag BIGINT, label TEXT) USING ZYRONLAKE"
        ),
    )
    .await
    .unwrap_or_else(|e| panic!("create {name}: {e}"));

    for file in 0..FILES {
        let base = file * ROWS_PER_FILE;
        let values: Vec<String> = (0..ROWS_PER_FILE)
            .map(|i| {
                let id = base + i;
                format!("({}, {}, 'label-{:08}')", id, id % TAGS, id)
            })
            .collect();
        exec_dml(
            server,
            &format!("INSERT INTO {name} VALUES {}", values.join(", ")),
        )
        .await;
    }
}

/// The table's newest manifest, which is where both sides of the cost
/// comparison come from
fn manifest_of(server: &Arc<ServerState>, table: &str) -> Arc<zyron_lake::ManifestFile> {
    let entry = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == table)
        .expect("table");
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).expect("lake log");
    log.manifest_at(log.latest_version()).expect("manifest")
}

fn first_ints(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    rows.iter()
        .map(|row| match row.first() {
            Some(ScalarValue::Int64(v)) => *v,
            Some(ScalarValue::Int32(v)) => *v as i64,
            other => panic!("expected an integer column, got {other:?}"),
        })
        .collect()
}

/// The cost comparison, on files a real writer produced.
///
/// Both directions in one table, so neither is a property of the data. One
/// manifest has to decline one index and take the other
#[tokio::test]
async fn test_the_index_cost_comparison_declines_a_resolved_key_and_takes_an_unresolved_one() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    table_with_a_clustered_and_a_shuffled_column(&server, &mut session, "ic").await;
    for column in ["id", "tag"] {
        exec_ddl(
            &server,
            &mut session,
            &format!("CREATE INDEX ic_{column}_ix ON ic ({column})"),
        )
        .await
        .expect("create index");
    }

    let manifest = manifest_of(&server, "ic");
    assert_eq!(
        manifest.entries.len(),
        FILES as usize,
        "one data file per insert"
    );
    let every_file: Vec<u64> = manifest.entries.iter().map(|e| e.partition_id).collect();

    // Bounds on id pick one file, so the scan reads one file's id column
    // and the index has nothing left to remove
    let id_spec = manifest.index_by_name("ix_id").expect("id index");
    let one_file = &every_file[..1];
    let scan_one = zyron_lake::scan_read_bytes(&manifest, one_file, id_spec.column_ids[0])
        .expect("the writer records per column sizes");
    let probe_id = zyron_lake::point_probe_read_bytes(
        &manifest,
        id_spec,
        Some(&zyron_lake::LakeValue::Int(7)),
    )
    .expect("probe cost");
    assert!(
        probe_id >= scan_one,
        "probing an index costs {probe_id} bytes against {scan_one} to read the key column of \
         the one file bounds admit, so the index must not be taken"
    );

    // Bounds on tag reject nothing, so the scan reads every file's tag
    // column and one index file replaces all of it
    let tag_spec = manifest.index_by_name("ix_tag").expect("tag index");
    let scan_all = zyron_lake::scan_read_bytes(&manifest, &every_file, tag_spec.column_ids[0])
        .expect("the writer records per column sizes");
    let probe_tag = zyron_lake::point_probe_read_bytes(
        &manifest,
        tag_spec,
        Some(&zyron_lake::LakeValue::Int(7)),
    )
    .expect("probe cost");
    assert!(
        probe_tag < scan_all,
        "probing an index costs {probe_tag} bytes against {scan_all} to read the same column out \
         of every file, so the index must be taken"
    );
}

/// A file's columns differ in width by more than an order of magnitude, so
/// the cost of reading one has to come from that column.
///
/// Charging a file's size over its column count makes a narrow key look as
/// expensive as the file's widest column, which is how an index gets taken
/// on a column whose bounds had already picked one file
#[tokio::test]
async fn test_reading_a_narrow_key_is_not_charged_for_a_wide_neighbour() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    table_with_a_clustered_and_a_shuffled_column(&server, &mut session, "nw").await;

    let manifest = manifest_of(&server, "nw");
    let entry = &manifest.entries[0];
    let columns = manifest.schema.columns.len() as u64;

    // Every column of a written file records what it costs to read
    let mut recorded = 0u64;
    for stat in entry.column_stats.iter() {
        recorded += stat
            .size_bytes
            .unwrap_or_else(|| panic!("column {} recorded no size", stat.column_id));
    }
    assert!(
        recorded < entry.size_bytes,
        "the segments are {recorded} bytes of a {} byte file, the rest being the header and \
         footer the reader also pays for",
        entry.size_bytes
    );

    let key = entry
        .stats_for(0)
        .expect("id stats")
        .size_bytes
        .expect("id size");
    let label = entry
        .stats_for(2)
        .expect("label stats")
        .size_bytes
        .expect("label size");
    assert!(
        key < label,
        "this table exists to have a wide column, and id is {key} bytes against label's {label}"
    );
    assert!(
        key < entry.size_bytes / columns,
        "reading id costs {key} bytes, below the {} byte per column average that a per file \
         average would have charged it",
        entry.size_bytes / columns
    );
}

/// A plan has to say which access path answered.
///
/// Without this the only visible difference between a scan whose statistics
/// were enough and one an index addressed is how long it took, which is
/// what left the question open in the first place
#[tokio::test]
async fn test_explain_analyze_reports_whether_an_index_answered_a_lake_scan() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    table_with_a_clustered_and_a_shuffled_column(&server, &mut session, "ex").await;

    // The answers before any index exists are the answers no index may change
    let by_id = first_ints(&query_values(&server, "SELECT id FROM ex WHERE id = 613").await);
    let by_tag = first_ints(&query_values(&server, "SELECT id FROM ex WHERE tag = 113").await);
    assert_eq!(by_id, vec![613]);
    assert_eq!(
        by_tag.len(),
        FILES as usize,
        "one row per file carries tag 113"
    );

    for column in ["id", "tag"] {
        exec_ddl(
            &server,
            &mut session,
            &format!("CREATE INDEX ex_{column}_ix ON ex ({column})"),
        )
        .await
        .expect("create index");
    }

    // Bounds on id already reach one file, so the scan answers it
    let resolved = analyze_lake_scan(&server, "SELECT id FROM ex WHERE id = 613").await;
    assert_eq!(
        resolved.aux[zyron_executor::operator::AUX_INDEX_FILES_READ],
        0,
        "an index that costs more than the scan it replaces must not be read"
    );
    assert_eq!(
        first_ints(&query_values(&server, "SELECT id FROM ex WHERE id = 613").await),
        by_id,
        "declining an index changed the answer"
    );

    // Bounds on tag reach every file, so the index answers it
    let addressed = analyze_lake_scan(&server, "SELECT id FROM ex WHERE tag = 113").await;
    assert!(
        addressed.aux[zyron_executor::operator::AUX_INDEX_FILES_READ] > 0,
        "the only thing that can narrow a predicate on tag is the index, and the plan says none \
         was read"
    );
    assert_eq!(
        addressed.aux[zyron_executor::operator::AUX_INDEX_ROWS_ADDRESSED],
        FILES as u64,
        "the index addressed the rows the predicate matches, one per file"
    );
    assert_eq!(
        first_ints(&query_values(&server, "SELECT id FROM ex WHERE tag = 113").await),
        by_tag,
        "answering through an index changed the answer"
    );

    // The rendered plan carries it too, which is where a person reads it
    let text = render_analyzed(&server, "SELECT id FROM ex WHERE tag = 113").await;
    assert!(text.contains("index_files_read="), "{text}");
    assert!(text.contains("index_rows_addressed=4"), "{text}");
}
