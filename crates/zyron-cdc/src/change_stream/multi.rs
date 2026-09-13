//! The read boundary of a stream over more than one table.
//!
//! A stream over several tables holds one position per table and reads to one
//! boundary across them. Every source reads to the newest change it holds,
//! and what keeps a transaction that touched two sources whole is the cut a
//! reader applies below the first change of any transaction that has not
//! ended: a committed transaction's changes are all at or below every
//! source's newest, so a window that reaches the newest takes all of them,
//! and one that stops below an open transaction's first change takes none.
//!
//! A source that stops changing does not hold the others back. Its window
//! ends at its own newest change, and the others end at theirs.

use std::collections::HashMap;

use zyron_catalog::ChangeStreamEntry;
use zyron_common::Result;

use super::ChangeStreamRuntime;

/// The version each source reads to, which is its own newest change.
///
/// `ended` says whether a transaction is over. The lowest version any
/// transaction still open wrote at, across every source, caps every window
/// just below it, so a transaction is either wholly inside the read or
/// wholly outside it.
pub fn boundary(
    runtime: &ChangeStreamRuntime,
    entry: &ChangeStreamEntry,
) -> Result<HashMap<u32, u64>> {
    boundary_below(runtime, entry, &|_| true)
}

/// The boundary with open transactions held out, see [`boundary`].
pub fn boundary_below(
    runtime: &ChangeStreamRuntime,
    entry: &ChangeStreamEntry,
    ended: &dyn Fn(u64) -> bool,
) -> Result<HashMap<u32, u64>> {
    let tables = entry.source.table_ids();
    for table_id in &tables {
        runtime.feed_on(*table_id, entry.branch)?;
    }
    let sources: Vec<(u32, Option<u64>)> = tables
        .iter()
        .map(|table_id| (*table_id, entry.branch))
        .collect();
    let held: Vec<(u32, crate::FeedBoundary)> = runtime
        .feeds()
        .boundaries(&sources, ended)
        .into_iter()
        .map(|((table_id, _), boundary)| (table_id, boundary))
        .collect();
    let cut = held
        .iter()
        .filter_map(|(_, boundary)| boundary.first_open)
        .min();
    let mut out = HashMap::with_capacity(tables.len());
    for (table_id, boundary) in held {
        let to = match cut {
            Some(first_open) => boundary.latest.min(first_open.saturating_sub(1)),
            None => boundary.latest,
        };
        out.insert(table_id, to.max(entry.position_of(table_id)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_feed::{CdfRegistry, ChangeRecord, ChangeType};
    use std::sync::Arc;
    use tempfile::TempDir;
    use zyron_catalog::{
        ChangeStreamMode, ChangeStreamOrigin, ChangeStreamSource, DatabaseId, SchemaId,
        StreamPosition,
    };

    fn record(table_id: u32, version: u64) -> ChangeRecord {
        ChangeRecord {
            change_type: ChangeType::Insert,
            commit_version: version,
            commit_timestamp: version as i64,
            table_id,
            txn_id: version,
            change_ordinal: 0,
            schema_version: 1,
            row_data: vec![9],
            primary_key_data: Vec::new(),
            is_last_in_txn: true,
            projected: false,
        }
    }

    fn multi_entry(tables: &[u32], positions: &[(u32, u64)]) -> ChangeStreamEntry {
        ChangeStreamEntry {
            id: 1,
            catalog_id: DatabaseId(1),
            schema_id: SchemaId(1),
            name: "gold".into(),
            source: ChangeStreamSource::Tables(tables.to_vec()),
            position: positions
                .iter()
                .map(|(table_id, version)| StreamPosition {
                    table_id: *table_id,
                    version: *version,
                    consumed: 0,
                })
                .collect(),
            created_at: 0,
            created_from: ChangeStreamOrigin::Now,
            mode: ChangeStreamMode::Standard,
            predicate: None,
            columns: None,
            owner_id: 1,
            last_advanced_at: 0,
            last_advanced_by: 0,
            stale: false,
            stale_reason: String::new(),
            needs_attention: false,
            attention_reason: String::new(),
            initial_rows_pending: false,
            branch: None,
        }
    }

    #[test]
    fn test_every_source_reads_to_its_own_newest_change() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = ChangeStreamRuntime::new(Arc::new(CdfRegistry::new(tmp.path().to_path_buf())));
        let fact = rt.feeds().enable_for_table(1, 7).expect("enables");
        let dim = rt.feeds().enable_for_table(2, 7).expect("enables");

        // One transaction touched both at version 5
        fact.append_batch(&[record(1, 5)]).expect("appends");
        dim.append_batch(&[record(2, 5)]).expect("appends");
        // The fact ran on to version 9 alone
        fact.append_batch(&[record(1, 9)]).expect("appends");

        let entry = multi_entry(&[1, 2], &[(1, 0), (2, 0)]);
        let bound = boundary(&rt, &entry).expect("resolves");
        assert_eq!(bound.get(&1), Some(&9), "the fact reads to its own newest");
        assert_eq!(bound.get(&2), Some(&5));

        let plan = rt.plan_read(&entry, None).expect("plans");
        let rows = rt.read(&plan).expect("reads");
        assert_eq!(
            rows.len(),
            3,
            "both halves of the shared commit and the later one"
        );
        assert!(rows.iter().any(|(t, _)| *t == 1));
        assert!(rows.iter().any(|(t, _)| *t == 2));
    }

    #[test]
    fn test_an_open_transaction_holds_every_source_below_its_first_change() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = ChangeStreamRuntime::new(Arc::new(CdfRegistry::new(tmp.path().to_path_buf())));
        let fact = rt.feeds().enable_for_table(1, 7).expect("enables");
        let dim = rt.feeds().enable_for_table(2, 7).expect("enables");

        // Transaction 5 committed on both. Transaction 7 wrote the dimension
        // at version 7 and is still open. Transaction 9 committed on the
        // fact after it
        fact.append_batch(&[record(1, 5)]).expect("appends");
        dim.append_batch(&[record(2, 5)]).expect("appends");
        dim.append_batch(&[record(2, 7)]).expect("appends");
        fact.append_batch(&[record(1, 9)]).expect("appends");

        let entry = multi_entry(&[1, 2], &[(1, 0), (2, 0)]);
        let open = |txn_id: u64| txn_id != 7;
        let bound = boundary_below(&rt, &entry, &open).expect("resolves");
        assert_eq!(
            bound.get(&1),
            Some(&6),
            "the fact stops below the open transaction"
        );
        assert_eq!(
            bound.get(&2),
            Some(&6),
            "and so does the dimension, which holds it"
        );
    }

    #[test]
    fn test_a_source_with_no_changes_does_not_stall_the_others() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = ChangeStreamRuntime::new(Arc::new(CdfRegistry::new(tmp.path().to_path_buf())));
        let busy = rt.feeds().enable_for_table(3, 7).expect("enables");
        let _quiet = rt.feeds().enable_for_table(4, 7).expect("enables");
        busy.append_batch(&[record(3, 4)]).expect("appends");

        let entry = multi_entry(&[3, 4], &[(3, 0), (4, 0)]);
        let bound = boundary(&rt, &entry).expect("resolves");
        assert_eq!(bound.get(&3), Some(&4));
        assert_eq!(bound.get(&4), Some(&0));

        let plan = rt.plan_read(&entry, None).expect("plans");
        assert_eq!(rt.read(&plan).expect("reads").len(), 1);
        // Both positions move in one advance, which is one catalog entry
        // rather than one write per source, so nothing can land between them
        assert_eq!(plan.advance_to(), vec![(3, 4), (4, 0)]);
    }
}
