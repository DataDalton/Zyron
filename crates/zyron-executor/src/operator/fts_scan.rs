//! Full-text search scan operator.
//!
//! Uses a fulltext inverted index to find matching documents by relevance
//! score. Fetches heap tuples by converting DocId back to TupleId and
//! appends a relevance score column to each output batch.

use std::sync::Arc;

use zyron_catalog::IndexId;
use zyron_common::Result;
use zyron_common::ZyronError;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_parser::ast::LiteralValue;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_search::{Bm25Scorer, FtsQueryParser, SimpleAnalyzer};
use zyron_storage::{HeapPage, TupleId};

use crate::batch::{create_builders, decode_tuple_into_builders, finalize_builders};
use crate::context::ExecutionContext;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Operator that executes a full-text search query against an inverted index,
/// then fetches matching heap tuples ordered by relevance score.
pub struct FulltextScanOperator {
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    output_columns: Vec<LogicalColumn>,
    /// Pre-computed (doc_id, score) results from the FTS index.
    results: Vec<(u64, f64)>,
    /// Current position in the results vector.
    cursor: usize,
    finished: bool,
}

impl FulltextScanOperator {
    /// Creates a new FTS scan operator. Extracts the query string from the
    /// match_against expression, runs the search, and pre-collects results.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        index_id: IndexId,
        columns: Vec<LogicalColumn>,
        match_expr: BoundExpr,
    ) -> Result<Self> {
        // Privilege check: require FulltextSearch on the table
        ctx.check_search_privilege(zyron_auth::PrivilegeType::FulltextSearch, table_id.0)?;

        let fts_index = ctx
            .get_fts_index(index_id)
            .ok_or_else(|| ZyronError::FtsIndexNotFound(format!("IndexId({})", index_id.0)))?;

        // Extract query string from the match_against function args.
        // The last argument is the query string literal.
        let query_str = extract_query_string(&match_expr)?;

        // Parse and execute the FTS query
        let fts_query = FtsQueryParser::parse(&query_str)?;
        let analyzer = SimpleAnalyzer;
        let scorer = Bm25Scorer::default();
        let results = fts_index.search(&fts_query, &analyzer, &scorer, 10000)?;

        let _table_entry = ctx.get_table_entry(table_id)?;

        Ok(Self {
            ctx,
            table_id,
            output_columns: columns,
            results,
            cursor: 0,
            finished: false,
        })
    }
}

impl Operator for FulltextScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.ctx.check_cancelled()?;

            let batch_size = self.ctx.batch_size;
            let table_entry = self.ctx.get_table_entry(self.table_id)?;

            let mut builders = create_builders(&self.output_columns, batch_size);
            let mut scores: Vec<f64> = Vec::with_capacity(batch_size);
            let mut row_count = 0usize;

            while row_count < batch_size && self.cursor < self.results.len() {
                let (doc_id, score) = self.results[self.cursor];
                self.cursor += 1;

                // Convert DocId back to TupleId
                let (page_num, slot_id) = zyron_search::decode_doc_id(doc_id);

                let page_id = PageId::new(table_entry.heap_file_id, page_num);
                let tid = TupleId::new(page_id, slot_id);

                // Read heap page and extract tuple
                let page_data: [u8; PAGE_SIZE] = self.ctx.disk_manager.read_page(page_id).await?;
                let page = HeapPage::from_bytes(page_data);

                let slot = zyron_storage::SlotId(tid.slot_id);
                let Some(tuple) = page.get_tuple(slot) else {
                    continue;
                };

                if tuple.is_deleted() {
                    continue;
                }
                if !tuple.header().is_visible_to(&self.ctx.snapshot) {
                    continue;
                }

                decode_tuple_into_builders(tuple.data(), &table_entry.columns, &mut builders);
                scores.push(score);
                row_count += 1;
            }

            if row_count == 0 {
                self.finished = true;
                return Ok(None);
            }

            let mut batch = finalize_builders(builders);
            // Append relevance score as an additional Float64 column.
            batch.columns.push(crate::column::Column {
                data: crate::column::ColumnData::Float64(scores),
                nulls: crate::column::NullBitmap::none(row_count),
                type_id: zyron_common::TypeId::Float64,
            });
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

/// Extracts the query string from a match_against BoundExpr::Function.
/// The last argument is expected to be a string literal containing the search query.
fn extract_query_string(expr: &BoundExpr) -> Result<String> {
    match expr {
        BoundExpr::Function { args, .. } => {
            // Last arg should be the query string
            if let Some(last) = args.last() {
                match last {
                    BoundExpr::Literal {
                        value: LiteralValue::String(s),
                        ..
                    } => Ok(s.clone()),
                    _ => Err(ZyronError::FtsQueryError(
                        "MATCH AGAINST query must be a string literal".to_string(),
                    )),
                }
            } else {
                Err(ZyronError::FtsQueryError(
                    "MATCH AGAINST requires a query argument".to_string(),
                ))
            }
        }
        _ => Err(ZyronError::FtsQueryError(
            "expected match_against function expression".to_string(),
        )),
    }
}
