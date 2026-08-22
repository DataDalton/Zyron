//! Row locking operator for SELECT ... FOR UPDATE/SHARE.
//!
//! Sits directly above the locked table's row-producing subtree, below the
//! projection, so every row it emits still carries a storage locator. Locks
//! are keyed on RowLocator, so heap resident and columnar resident rows lock
//! uniformly. NOWAIT fails on the first contended row, SKIP LOCKED filters
//! contended rows out of the batch, the default waits for the holder and
//! then verifies the row's latest committed state still matches what this
//! snapshot returned, failing with a conflict when the released holder
//! committed a change (first committer wins, no stale image is handed to
//! the application under a lock).

use std::sync::Arc;

use zyron_common::{Result, RowLocator, ZyronError};
use zyron_planner::binder::{RowLockMode, RowLockWait};
use zyron_storage::LockMode;

use crate::context::ExecutionContext;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

pub struct LockRowsOperator {
    child: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    mode: LockMode,
    wait: RowLockWait,
    /// Rows still allowed to lock and emit, seeded from the plan's literal
    /// LIMIT plus OFFSET. Keeps FOR UPDATE SKIP LOCKED LIMIT n locking
    /// exactly n rows instead of a whole batch. None means unbounded
    remaining: Option<u64>,
}

impl LockRowsOperator {
    pub fn new(
        child: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        mode: RowLockMode,
        wait: RowLockWait,
        cap: Option<u64>,
    ) -> Self {
        let mode = match mode {
            RowLockMode::Exclusive => LockMode::Exclusive,
            RowLockMode::Shared => LockMode::Shared,
        };
        Self {
            child,
            ctx,
            table_id,
            mode,
            wait,
            remaining: cap,
        }
    }
}

impl Operator for LockRowsOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.remaining == Some(0) {
                return Ok(None);
            }
            let Some(locks) = self.ctx.row_locks.clone() else {
                return Err(ZyronError::ExecutionError(
                    "FOR UPDATE/SHARE requires the session lock table".to_string(),
                ));
            };
            let txn_id = self.ctx.txn_id as u64;
            let table_id = self.table_id.0;
            loop {
                let Some(eb) = self.child.next().await? else {
                    return Ok(None);
                };
                let batch = eb.batch;
                let Some(mut locators) = eb.locators else {
                    return Err(ZyronError::ExecutionError(
                        "row locking received a batch without row locators".to_string(),
                    ));
                };
                match self.wait {
                    RowLockWait::Nowait | RowLockWait::Wait => {
                        let take = self
                            .remaining
                            .map(|r| (r as usize).min(locators.len()))
                            .unwrap_or(locators.len());
                        for loc in &locators[..take] {
                            if self.wait == RowLockWait::Nowait {
                                locks
                                    .lock_row_or_holder(txn_id, table_id, *loc, self.mode)
                                    .map_err(|holder| ZyronError::TransactionConflict {
                                        txn_id,
                                        reason: format!(
                                            "could not lock row {loc}, held by txn {holder} (NOWAIT)"
                                        ),
                                    })?;
                            } else if locks
                                .lock_row_or_holder(txn_id, table_id, *loc, self.mode)
                                .is_err()
                            {
                                locks
                                    .lock_row_wait(txn_id, table_id, *loc, self.mode)
                                    .await?;
                                if row_changed_since_snapshot(&self.ctx, self.table_id, *loc)
                                    .await?
                                {
                                    return Err(ZyronError::TransactionConflict {
                                        txn_id,
                                        reason: format!(
                                            "row {loc} was changed by a concurrently committed \
                                             transaction, retry the transaction"
                                        ),
                                    });
                                }
                            }
                        }
                        if let Some(r) = self.remaining.as_mut() {
                            *r -= take as u64;
                        }
                        if take < locators.len() {
                            locators.truncate(take);
                            let batch = batch.slice(0, take);
                            return Ok(Some(ExecutionBatch {
                                batch,
                                locators: Some(locators),
                            }));
                        }
                        return Ok(Some(ExecutionBatch {
                            batch,
                            locators: Some(locators),
                        }));
                    }
                    RowLockWait::SkipLocked => {
                        let cap = self.remaining.map(|r| r as usize).unwrap_or(usize::MAX);
                        let mut keep: Vec<u32> = Vec::with_capacity(locators.len());
                        for (i, loc) in locators.iter().enumerate() {
                            if keep.len() >= cap {
                                break;
                            }
                            if locks.try_lock_row(txn_id, table_id, *loc, self.mode) {
                                keep.push(i as u32);
                            }
                        }
                        if keep.is_empty() {
                            // every candidate row is locked elsewhere, pull
                            // the next batch
                            continue;
                        }
                        if let Some(r) = self.remaining.as_mut() {
                            *r -= keep.len() as u64;
                        }
                        if keep.len() == locators.len() {
                            return Ok(Some(ExecutionBatch {
                                batch,
                                locators: Some(locators),
                            }));
                        }
                        let kept: Vec<RowLocator> =
                            keep.iter().map(|&i| locators[i as usize]).collect();
                        let batch = batch.take(&keep);
                        return Ok(Some(ExecutionBatch {
                            batch,
                            locators: Some(kept),
                        }));
                    }
                }
            }
        })
    }
}

/// After a contended wait completes, checks whether the row's latest
/// committed state still matches what this snapshot returned. A committed
/// deleter on a heap tuple, a pruned slot, or a committed-but-invisible
/// columnar patch or supersede all mean the returned image is stale. Free
/// function so the operator's future stays Send, the boxed child operator
/// is not Sync
async fn row_changed_since_snapshot(
    ctx: &Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    locator: RowLocator,
) -> Result<bool> {
    let snapshot = &ctx.snapshot;
    let own_txn = ctx.txn_id as u64;
    match locator {
        RowLocator::Heap { page, slot } => {
            let data = crate::operator::scan::read_page_through_pool(
                &ctx.buffer_pool,
                &ctx.disk_manager,
                page,
            )
            .await?;
            let Some(view) = zyron_storage::HeapPage::get_tuple_view_from_slice(
                &data,
                zyron_storage::SlotId(slot),
            ) else {
                // slot pruned or reused, the returned image no longer exists
                return Ok(true);
            };
            let xmax = view.header.xmax as u64;
            Ok(xmax != 0 && xmax != own_txn && snapshot.status_map().is_committed(xmax))
        }
        RowLocator::Columnar { file_id, sys_rowid } => {
            let te = ctx.catalog.get_table_by_id(table_id)?;
            let store = crate::operator::modify::columnar_patch_store(&te)?;
            let overlay = match ctx.active_branch_id {
                Some(branch) => store.row_overlay_on(branch, file_id, sys_rowid),
                None => store.row_overlay(file_id, sys_rowid),
            };
            let Some(overlay) = overlay else {
                return Ok(false);
            };
            // a supersede or patch from a txn that committed but is not
            // visible to this snapshot is a concurrent change
            let concurrent = |xid: u64| {
                xid != own_txn
                    && snapshot.status_map().is_committed(xid)
                    && !snapshot.is_visible(xid, 0)
            };
            if overlay.supersedes.iter().any(|&x| concurrent(x)) {
                return Ok(true);
            }
            Ok(overlay
                .patches
                .values()
                .any(|chain| chain.iter().any(|p| concurrent(p.patch_xid))))
        }
        // no lake DML path exists, a locked lake row cannot have been
        // rewritten while this transaction waited
        RowLocator::Lake { .. } => Ok(false),
    }
}
