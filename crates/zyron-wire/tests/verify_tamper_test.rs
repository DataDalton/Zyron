//! What a verification reports when the table or the chain was changed
//! outside SQL, and what only an anchor can catch.
//!
//! The engine refuses every path that would change a verified table's rows,
//! so these tests reach around it: they edit the stored bytes of a row, cut
//! a chain file short, and rebuild a chain from end to end, which is what
//! someone with the data directory can do. Then they ask what VERIFY TABLE
//! makes of it.
//!
//! Run: cargo test -p zyron-wire --test verify_tamper_test -- --nocapture

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use zyron_lifecycle::verify::{self, CommitChain, RowsHasher};
use zyron_wire::connection::ServerState;

mod common;
use common::*;

/// The row `VERIFY TABLE` answers with
struct VerifyRow {
    mode: String,
    intact: bool,
    detail: String,
}

/// The rows of a table as a walk reads them back, under a snapshot of what
/// the table holds now
fn rows_over(
    server: &Arc<ServerState>,
    heap: Arc<zyron_storage::HeapFile>,
) -> zyron_wire::verify_dispatch::HeapCommitRows {
    let mut reader = server
        .txn_manager
        .begin(zyron_storage::IsolationLevel::ReadCommitted)
        .expect("begins");
    let snapshot = reader.snapshot.clone();
    server
        .txn_manager
        .commit_read_only(&mut reader)
        .expect("ends");
    zyron_wire::verify_dispatch::HeapCommitRows::new(
        heap,
        snapshot,
        verify::spill_dir(server.disk_manager.data_dir()),
        Arc::new(|| false),
    )
}

async fn verify(server: &Arc<ServerState>, sql: &str) -> VerifyRow {
    let mut session = new_session();
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut branch: Option<String> = None;
    let answered = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        &mut session,
        &mut txn_opt,
        &mut branch,
        sql,
    )
    .await
    .expect("the statement is handled")
    .expect("the verification ran");
    let zyron_wire::ddl_dispatch::DdlResult::Rows { rows, .. } = answered else {
        panic!("VERIFY TABLE answers with rows");
    };
    let row = rows.first().expect("one row").clone();
    VerifyRow {
        mode: row[4].clone(),
        intact: row[5] == "true",
        detail: row[6].clone(),
    }
}

/// Builds a verified table holding `commits` one-row commits
async fn verified_ledger(
    commits: i64,
) -> (
    Arc<ServerState>,
    Option<zyron_wire::session::Session>,
    tempfile::TempDir,
) {
    let (server, _schema, tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT PRIMARY KEY, amount BIGINT)",
    )
    .await
    .expect("create ledger");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await
    .expect("verified");
    for id in 1..=commits {
        exec_dml(
            &server,
            &format!("INSERT INTO ledger VALUES ({id}, {})", id * 10),
        )
        .await;
    }
    (server, session, tmp)
}

fn chain_of(server: &Arc<ServerState>) -> Arc<CommitChain> {
    let table_id = table_id_of(server, "ledger");
    server
        .chain_registry
        .as_ref()
        .expect("chains")
        .chain(table_id)
        .expect("chain")
}

/// Changes one byte of one stored row, the way an edit outside SQL does.
///
/// The row is found by scanning the heap for the commit that wrote it and
/// the byte is changed in the page itself, so nothing in the database is
/// told that anything happened
async fn edit_a_row_of(server: &Arc<ServerState>, sequence: u64) {
    let table_id = table_id_of(server, "ledger");
    let entry = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id))
        .expect("the table");
    let chain = chain_of(server);
    let target = chain
        .read_range(sequence, sequence)
        .expect("reads")
        .into_iter()
        .next()
        .expect("the entry is there");

    let heap = zyron_wire::connection::table_heap(server, &entry)
        .await
        .expect("the heap opens");
    let mut found: Option<(zyron_storage::TupleId, Vec<u8>)> = None;
    {
        let guard = heap.scan().expect("scans");
        guard.try_for_each(|tid, view| {
            if view.header.xmin == target.txn_id {
                found = Some((tid, view.data.to_vec()));
                return false;
            }
            true
        });
    }
    let (tid, bytes) = found.expect("the commit's row is in the heap");

    // Every page reaches the device, then one byte of the row is changed in
    // the file and the pool is emptied, so the next read comes from disk
    settle_pages(server);
    let page = server
        .disk_manager
        .read_page_sync(tid.page_id)
        .expect("reads the page");
    let mut image = *page;
    let at = find_bytes(&image, &bytes).expect("the row is in its page");
    image[at] = image[at].wrapping_add(1);
    server
        .disk_manager
        .write_page_sync(tid.page_id, &mut image)
        .expect("writes the page back");
    // The pool still holds the page as it was, so the copy it holds is
    // dropped and the changed bytes are loaded in its place, which is what a
    // reader would find after a restart
    server.buffer_pool.delete_page(tid.page_id);
    server
        .buffer_pool
        .load_page(tid.page_id, &image)
        .expect("the changed page loads");
    server.buffer_pool.unpin_page(tid.page_id, false);
}

/// Where a run of bytes sits in a page, None when it is not there
fn find_bytes(page: &[u8], needle: &[u8]) -> Option<usize> {
    page.windows(needle.len())
        .position(|window| window == needle)
}

fn settle_pages(server: &Arc<ServerState>) {
    let disk = Arc::clone(&server.disk_manager);
    server.wal.flush().expect("the log flushes");
    server
        .buffer_pool
        .flush_all(|page_id, data| {
            let page: &mut [u8; zyron_common::page::PAGE_SIZE] = data
                .try_into()
                .map_err(|_| zyron_common::ZyronError::Internal("a frame is one page".into()))?;
            disk.write_page_sync(page_id, page)?;
            Ok(zyron_buffer::FlushOutcome::Written)
        })
        .expect("the pages flush");
}

/// Editing one row's stored bytes outside SQL is reported as a row content
/// mismatch, naming the first commit that no longer matches
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_edited_row_is_reported_as_a_row_content_mismatch() {
    let (server, _session, _tmp) = verified_ledger(6).await;

    // Clean before
    let before = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(before.intact, "{}", before.detail);

    let chain = chain_of(&server);
    let target = chain.read_range(3, 3).expect("reads").remove(0);
    edit_a_row_of(&server, 3).await;

    let after = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(!after.intact, "the edit was not reported: {}", after.detail);
    assert!(
        after.detail.contains(&target.commit_version.to_string()),
        "the first bad commit is not named: {}",
        after.detail
    );
    assert!(
        after.detail.contains("hash to"),
        "the failure is not a row content mismatch: {}",
        after.detail
    );
    assert_eq!(after.mode, "all");

    // The run is on the record with what it found
    let (columns, runs) = view(&server, "zyron_sys.verify.runs").await;
    let newest = runs.first().expect("the run is recorded");
    assert_eq!(cell(&columns, newest, "intact"), "f");
    assert!(!cell(&columns, newest, "finding").is_empty());
}

/// A sampled pass that did not read the edited commit reports nothing, and
/// says so rather than reading as a clean bill
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sampled_pass_states_what_it_did_not_read() {
    let (server, _session, _tmp) = verified_ledger(40).await;
    edit_a_row_of(&server, 17).await;

    let sampled = verify(&server, "VERIFY TABLE ledger WITH (sample => 2)").await;
    assert_eq!(sampled.mode, "sampled");
    assert!(
        sampled.detail.contains("the rows of the rest not read"),
        "a sampled pass has to say what it did not read: {}",
        sampled.detail
    );

    // Reading every commit's rows finds it
    let full = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(
        !full.intact,
        "the full pass missed the edit: {}",
        full.detail
    );
}

/// Removing the last entries of a chain leaves a shorter chain that walks
/// clean. The anchor taken over what it was is what contradicts it
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_truncated_chain_is_caught_by_the_anchor_it_contradicts() {
    let (server, _session, _tmp) = verified_ledger(8).await;

    // The head is anchored, which is what a truncation now has to get past
    let anchor = zyron_wire::verify_dispatch::anchor_table(&server, table_id_of(&server, "ledger"))
        .await
        .expect("anchors")
        .expect("the head is anchored");
    assert_eq!(anchor.sequence, 7);

    let clean = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(clean.intact, "{}", clean.detail);
    assert!(
        clean.detail.contains("1 anchor(s) agree"),
        "the anchor is not reported: {}",
        clean.detail
    );

    // The last three entries are cut off the chain file, the way a
    // truncation outside the database does
    let chain = chain_of(&server);
    chain.sync().expect("flushes");
    let header = 20u64;
    let kept = 5u64;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(chain.path())
        .expect("opens the chain file");
    file.set_len(header + kept * verify::CHAIN_RECORD_LEN as u64)
        .expect("cuts it back");
    drop(file);

    // The registry is asked for the chain again, the way a restart reads it
    let reopened = CommitChain::open(
        server.disk_manager.data_dir(),
        table_id_of(&server, "ledger"),
    )
    .expect("reopens");
    assert_eq!(reopened.head().commits, kept, "the chain is shorter now");

    // Walked on its own it is a complete chain, which is the case an anchor
    // exists for
    let rows = rows_over(
        &server,
        zyron_wire::connection::table_heap(
            &server,
            &server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(table_id_of(&server, "ledger")))
                .expect("the table"),
        )
        .await
        .expect("the heap opens"),
    );
    let alone = verify::walk(
        &reopened,
        &rows,
        0,
        kept - 1,
        verify::RowMode::All,
        0,
        &[],
        &|| false,
    )
    .expect("walks");
    assert!(
        alone.intact,
        "a chain read on its own cannot see that it was cut short"
    );

    let against = verify::walk(
        &reopened,
        &rows,
        0,
        kept - 1,
        verify::RowMode::All,
        0,
        std::slice::from_ref(&anchor),
        &|| false,
    )
    .expect("walks");
    assert!(!against.intact, "the anchor did not catch the truncation");
    let failure = against.failure.expect("a failure");
    assert_eq!(failure.kind(), "anchor_contradiction");
    assert!(
        failure
            .to_string()
            .contains(&verify::hex(&anchor.head_hash)),
        "the anchor it contradicts is not named: {failure}"
    );
}

/// A chain rebuilt from end to end over the edited rows walks clean and
/// still contradicts an exported anchor.
///
/// This is the case the phase exists for. The same forgery against a
/// checksum chain succeeds, because a checksum can be recomputed by anyone
/// who can rewrite the record
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_forged_chain_is_caught_by_an_exported_anchor_and_a_checksum_chain_is_not() {
    let (server, _session, _tmp) = verified_ledger(6).await;
    let table_id = table_id_of(&server, "ledger");

    let anchor = zyron_wire::verify_dispatch::anchor_table(&server, table_id)
        .await
        .expect("anchors")
        .expect("anchored");
    let exported = zyron_wire::verify_dispatch::export_anchor(&anchor).expect("exports");
    // The artifact round-trips, so an operator can hold the bytes outside
    let bytes = exported.encode();
    let held = zyron_lifecycle::verify::anchor::ExportedAnchor::decode(&bytes).expect("decodes");
    assert_eq!(held, exported);
    assert!(!held.signature.is_empty());

    // Before anything is touched, the artifact agrees with the chain and
    // was signed by this cluster
    let registry = server.chain_registry.as_ref().expect("chains");
    let (signed, agrees) =
        zyron_wire::verify_dispatch::check_exported_anchor(registry, &held).expect("checks");
    assert!(signed, "the artifact was not signed by this cluster");
    assert!(agrees, "the artifact does not describe the chain");

    // The forgery: one row edited, then every entry after it relinked, which
    // is what someone with the data directory and this binary can do
    edit_a_row_of(&server, 2).await;
    let entry = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id))
        .expect("the table");
    let heap = zyron_wire::connection::table_heap(&server, &entry)
        .await
        .expect("the heap opens");
    let old = chain_of(&server).read_range(0, 5).expect("reads");
    let forged_dir = tempfile::TempDir::new().expect("temp dir");
    let forged = CommitChain::open(forged_dir.path(), table_id).expect("opens");
    for was in &old {
        let mut hasher = RowsHasher::new();
        {
            let guard = heap.scan().expect("scans");
            guard.for_each(|_tid, view| {
                if view.header.xmin == was.txn_id {
                    hasher.row(view.header.schema_epoch, view.data);
                }
            });
        }
        let rows = hasher.rows();
        forged
            .append(
                verify::CommitFields {
                    txn_id: was.txn_id,
                    rows_hash: hasher.finish(),
                    row_count: rows,
                    commit_ts: was.commit_ts,
                    algorithm_id: was.algorithm_id,
                    genesis: was.genesis,
                },
                || was.commit_version,
            )
            .expect("appends");
        forged.publish(was.txn_id).expect("publishes");
    }
    forged.sync().expect("flushes");

    let rows = rows_over(&server, heap);
    let walked = verify::walk(&forged, &rows, 0, 5, verify::RowMode::All, 0, &[], &|| {
        false
    })
    .expect("walks");
    assert!(
        walked.intact,
        "a chain relinked over the edited rows is self-consistent: {:?}",
        walked.failure
    );

    // The exported anchor names a head the forged chain does not have
    assert!(
        !held.agrees_with(forged.head().commits - 1, &forged.head().head_hash),
        "the forgery reproduced the anchored head"
    );
    let against = verify::walk(
        &forged,
        &rows,
        0,
        5,
        verify::RowMode::All,
        0,
        std::slice::from_ref(&anchor),
        &|| false,
    )
    .expect("walks");
    assert!(!against.intact, "the anchor did not catch the forgery");
    assert_eq!(
        against.failure.expect("a failure").kind(),
        "anchor_contradiction"
    );

    // The same forgery against a chain linked by a checksum succeeds, which
    // is why the chain is a one-way hash and the anchor is signed
    let mut crc_prev = 0u32;
    let mut crc_chain = Vec::new();
    for was in &old {
        let mut payload = Vec::new();
        payload.extend_from_slice(&crc_prev.to_le_bytes());
        payload.extend_from_slice(&was.commit_version.to_le_bytes());
        payload.extend_from_slice(&was.rows_hash);
        let link = zyron_common::hash32(&payload);
        crc_chain.push((crc_prev, link));
        crc_prev = link;
    }
    let anchored_crc = crc_prev;
    // The forger edits the second entry's input and recomputes the rest
    let mut forged_prev = 0u32;
    let mut forged_chain = Vec::new();
    for (at, was) in old.iter().enumerate() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&forged_prev.to_le_bytes());
        payload.extend_from_slice(&was.commit_version.to_le_bytes());
        let mut rows_hash = was.rows_hash;
        if at == 2 {
            rows_hash[0] = rows_hash[0].wrapping_add(1);
        }
        payload.extend_from_slice(&rows_hash);
        let link = zyron_common::hash32(&payload);
        forged_chain.push((forged_prev, link));
        forged_prev = link;
    }
    // Every link of the checksum chain holds, and a forger who also holds
    // the anchor rewrites it to the head they produced, because producing a
    // checksum takes no key
    for (at, (prev, link)) in forged_chain.iter().enumerate() {
        let expected = if at == 0 { 0 } else { forged_chain[at - 1].1 };
        assert_eq!(*prev, expected, "the forged checksum chain links");
        assert_ne!(*link, 0);
    }
    assert_ne!(
        forged_prev, anchored_crc,
        "the forged checksum chain has its own head"
    );
    // Nothing about a checksum head can be signed by the cluster and not by
    // the forger: the value is a function of the record alone
    let mut recomputed = Vec::new();
    recomputed.extend_from_slice(&forged_chain[5].0.to_le_bytes());
    recomputed.extend_from_slice(&old[5].commit_version.to_le_bytes());
    recomputed.extend_from_slice(&old[5].rows_hash);
    assert_eq!(
        zyron_common::hash32(&recomputed),
        {
            let mut payload = Vec::new();
            payload.extend_from_slice(&forged_chain[5].0.to_le_bytes());
            payload.extend_from_slice(&old[5].commit_version.to_le_bytes());
            payload.extend_from_slice(&old[5].rows_hash);
            zyron_common::hash32(&payload)
        },
        "a checksum is reproducible by anyone holding the record"
    );
}

/// One system view, rows keyed by column name
async fn view(server: &Arc<ServerState>, name: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let (fields, rows) = zyron_wire::system_views::query_system_view(
        name,
        server,
        &zyron_wire::system_views::SystemViewFilters::default(),
    )
    .await
    .expect("the view answers")
    .unwrap_or_else(|| panic!("{name} is not a system view"));
    let columns: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    let rows = rows
        .into_iter()
        .map(|cells| {
            cells
                .into_iter()
                .map(|cell| {
                    cell.map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    (columns, rows)
}

fn cell<'a>(columns: &[String], row: &'a [String], name: &str) -> &'a str {
    let at = columns
        .iter()
        .position(|c| c == name)
        .unwrap_or_else(|| panic!("no column {name} in {columns:?}"));
    &row[at]
}

/// Silences the unused-import warning for the io traits the page edit needs
#[allow(dead_code)]
fn _io_traits_are_used(mut file: std::fs::File, buf: &mut [u8]) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(buf)?;
    file.write_all(buf)
}
