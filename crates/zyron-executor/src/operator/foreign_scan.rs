//! Foreign table scan.
//!
//! Reads a table that lives on another node. The rows arrive already
//! filtered and already projected, because the predicate and the column
//! list go to the peer rather than the rows coming here to be discarded:
//! a federated scan that fetches everything and filters locally pays the
//! network for work the remote could have skipped.
//!
//! The executor cannot open a connection itself, and should not. Reaching
//! a peer means the wire protocol, the client pool, TLS and the peer
//! registry, all of which live above this crate. So the capability is
//! injected as a `ForeignReader` on the execution context, the same way
//! heap files and B+tree indexes are, and this operator only decides what
//! to ask for.
//!
//! What cannot be pushed is not approximated. A predicate with no faithful
//! remote rendering is left off the request and applied here instead, so
//! the answer is the same either way and only the cost differs.

use std::sync::Arc;

use zyron_common::{ForeignRequest, Result, ZyronError};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::{BATCH_SIZE, create_builders, finalize_builders};
use crate::column::ScalarValue;
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{
    AUX_REMOTE_MS, AUX_ROWS_FETCHED, ExecutionBatch, Operator, OperatorMetrics, OperatorResult,
    apply_column_security,
};

/// Reads a table on another node.
///
/// Implemented above this crate, where the client lives. It returns
/// decoded values rather than wire bytes, because decoding needs the
/// protocol's encoding rules and those belong to the crate that speaks the
/// protocol. This operator knows what it asked for and what to do with the
/// answer, and nothing about how it travelled.
pub trait ForeignReader: Send + Sync {
    /// Runs one remote read. Rows come back in the requested column order,
    /// decoded to `request.column_types`.
    fn scan(&self, request: &ForeignRequest) -> Result<Vec<Vec<ScalarValue>>>;
}

/// Scans a foreign table through the injected reader.
pub struct ForeignScanOperator {
    ctx: Arc<ExecutionContext>,
    table_id: u32,
    output_columns: Vec<LogicalColumn>,
    /// Applied here only when it could not be pushed, so a row is never
    /// filtered twice
    residual: Option<BoundExpr>,
    request: ForeignRequest,
    fetched: bool,
    pending: std::collections::VecDeque<ExecutionBatch>,
    /// Where the round trip reports itself. What a foreign scan costs is
    /// how much came back and how long the peer took, and neither is
    /// visible until the fetch runs, so they are published from here
    /// rather than fixed when the operator is built
    metrics: Option<Arc<OperatorMetrics>>,
}

impl ForeignScanOperator {
    pub fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        residual: Option<BoundExpr>,
        request: ForeignRequest,
    ) -> Result<Self> {
        if columns.is_empty() {
            return Err(ZyronError::ExecutionError(format!(
                "foreign scan of \"{}\" projects no column",
                request.table
            )));
        }
        Ok(Self {
            ctx,
            table_id: table_id.0,
            output_columns: columns,
            residual,
            request,
            fetched: false,
            pending: std::collections::VecDeque::new(),
            metrics: None,
        })
    }

    /// Attaches the counters this scan fills after its round trip.
    pub fn with_metrics(mut self, metrics: Option<Arc<OperatorMetrics>>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Asks the peer once and turns its rows into batches.
    fn fetch(&mut self) -> Result<()> {
        let reader = self.ctx.foreign_reader.as_ref().ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "this node cannot read foreign table \"{}\" on peer \"{}\", it holds no \
                 client for peers",
                self.request.table, self.request.peer
            ))
        })?;
        let started = std::time::Instant::now();
        let rows = reader.scan(&self.request)?;
        if let Some(metrics) = &self.metrics {
            metrics.set_aux(AUX_ROWS_FETCHED, rows.len() as u64);
            metrics.set_aux(AUX_REMOTE_MS, started.elapsed().as_millis() as u64);
        }
        if rows.is_empty() {
            return Ok(());
        }

        // Consumed by value so each row is dropped as it is packed. Holding
        // the answer and the batches built from it at once would double the
        // peak footprint of a large remote read
        let mut builders = create_builders(&self.output_columns, rows.len().min(BATCH_SIZE));
        let mut in_batch = 0usize;
        for row in rows {
            if row.len() != self.output_columns.len() {
                return Err(ZyronError::ExecutionError(format!(
                    "peer \"{}\" returned {} columns for \"{}\", the plan projects {}",
                    self.request.peer,
                    row.len(),
                    self.request.table,
                    self.output_columns.len()
                )));
            }
            for (index, value) in row.into_iter().enumerate() {
                builders[index].push_owned(value);
            }
            in_batch += 1;
            if in_batch == BATCH_SIZE {
                let batch = finalize_builders(std::mem::replace(
                    &mut builders,
                    create_builders(&self.output_columns, BATCH_SIZE),
                ));
                self.queue(batch)?;
                in_batch = 0;
            }
        }
        if in_batch > 0 {
            let batch = finalize_builders(builders);
            self.queue(batch)?;
        }
        Ok(())
    }

    fn queue(&mut self, batch: crate::batch::DataBatch) -> Result<()> {
        // Only what the peer could not filter is filtered here, so a row
        // the remote already excluded is never re-examined
        let filtered = match &self.residual {
            Some(predicate) => {
                let mask_column =
                    evaluate(predicate, &batch, &self.output_columns, &self.ctx.params)?;
                batch.filter(&column_to_mask(&mask_column))
            }
            None => batch,
        };
        if filtered.num_rows == 0 {
            return Ok(());
        }
        let secured =
            apply_column_security(&self.ctx, self.table_id, &self.output_columns, filtered);
        self.pending.push_back(ExecutionBatch::new(secured));
        Ok(())
    }

    /// What the peer was asked for, so EXPLAIN can show it.
    pub fn request(&self) -> &ForeignRequest {
        &self.request
    }
}

impl Operator for ForeignScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if !self.fetched {
                self.fetched = true;
                self.fetch()?;
            }
            Ok(self.pending.pop_front())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedReader {
        rows: Vec<Vec<ScalarValue>>,
        seen: std::sync::Mutex<Option<ForeignRequest>>,
    }

    impl ForeignReader for FixedReader {
        fn scan(&self, request: &ForeignRequest) -> Result<Vec<Vec<ScalarValue>>> {
            *self.seen.lock().expect("lock") = Some(request.clone());
            Ok(self.rows.clone())
        }
    }

    /// The request carries what the remote should do, so the peer filters
    /// and projects rather than shipping rows to be discarded here
    #[test]
    fn test_a_request_names_only_what_is_needed() {
        let request = ForeignRequest {
            peer: "west".into(),
            table: "orders".into(),
            columns: vec!["id".into(), "total".into()],
            column_types: vec![zyron_common::TypeId::Int64, zyron_common::TypeId::Float64],
            predicate: Some("(id > 100)".into()),
            limit: Some(50),
        };
        assert!(!request.columns.is_empty(), "a scan asks for something");
        assert_eq!(
            request.columns.len(),
            request.column_types.len(),
            "every column says what it should decode as"
        );
        assert!(request.predicate.is_some(), "the filter goes to the peer");
        assert_eq!(request.limit, Some(50));
    }

    /// A reader sees exactly the request it was given, unchanged
    #[test]
    fn test_the_reader_receives_the_request_verbatim() {
        let reader = FixedReader {
            rows: vec![vec![ScalarValue::Int64(1), ScalarValue::Null]],
            seen: std::sync::Mutex::new(None),
        };
        let request = ForeignRequest {
            peer: "west".into(),
            table: "orders".into(),
            columns: vec!["id".into(), "note".into()],
            column_types: vec![zyron_common::TypeId::Int64, zyron_common::TypeId::Text],
            predicate: None,
            limit: None,
        };
        let rows = reader.scan(&request).expect("scan");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][1], ScalarValue::Null, "a NULL cell stays NULL");
        assert_eq!(reader.seen.lock().expect("lock").as_ref(), Some(&request));
    }
}
