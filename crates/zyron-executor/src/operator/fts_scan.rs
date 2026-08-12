//! Full-text search scan operator.
//!
//! Uses a fulltext inverted index to find matching documents by relevance
//! score. Result DocIds resolve through the document registry to row
//! locators, heap hits read their page, columnar hits come from the
//! pre-fetched batch, and a relevance score column is appended.

use std::sync::Arc;

use zyron_catalog::IndexId;
use zyron_common::Result;
use zyron_common::RowLocator;
use zyron_common::ZyronError;
use zyron_parser::ast::LiteralValue;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_search::{Bm25Scorer, FtsQueryParser, SimpleAnalyzer};
use zyron_storage::HeapPage;

use crate::batch::{
    build_column_to_builder_map, create_builders, decode_tuple_into_builders, finalize_builders,
};
use crate::context::ExecutionContext;
use crate::operator::doc_fetch::DocRowFetcher;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Operator that executes a full-text search query against an inverted index,
/// then fetches matching rows ordered by relevance score.
pub struct FulltextScanOperator {
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    output_columns: Vec<LogicalColumn>,
    /// Pre-computed (doc_id, score) results from the FTS index.
    results: Vec<(u64, f64)>,
    /// Resolved locators plus pre-fetched columnar hits, result-aligned.
    fetcher: DocRowFetcher,
    /// Current position in the results vector.
    cursor: usize,
    finished: bool,
    /// When set, heap hits are dated by commit LSN at this version instead of
    /// being resolved against the live snapshot
    heap_version: Option<u64>,
    /// Table and index IO counters. The search ran to completion before any
    /// row was fetched, so the entries it examined were recorded then
    io_stats: crate::operator::IndexScanStats,
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
        as_of: Option<zyron_planner::logical::AsOfTarget>,
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

        let docs: Vec<u64> = results.iter().map(|r| r.0).collect();
        let fetcher =
            DocRowFetcher::prepare(&ctx, table_id, &columns, &docs, as_of.as_ref()).await?;
        let io_stats = crate::operator::IndexScanStats::open(
            &ctx,
            table_id.0,
            index_id.0,
            results.len(),
        );

        Ok(Self {
            ctx,
            table_id,
            output_columns: columns,
            results,
            fetcher,
            cursor: 0,
            finished: false,
            heap_version: crate::operator::doc_fetch::heap_as_of_version(as_of.as_ref()),
            io_stats,
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
            let output_ids: Vec<zyron_catalog::ColumnId> =
                self.output_columns.iter().map(|c| c.column_id).collect();
            let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);
            let mut scores: Vec<f64> = Vec::with_capacity(batch_size);
            let mut row_count = 0usize;
            // Heap pages fetched to resolve this batch's hits, folded into
            // the table counters once when the batch is done
            let mut pages_read = 0u64;

            while row_count < batch_size && self.cursor < self.results.len() {
                let (_doc_id, score) = self.results[self.cursor];
                let locator = self.fetcher.locators[self.cursor];
                self.cursor += 1;

                match locator {
                    // dead document, deleted after the index entry was scored
                    None => continue,
                    Some(RowLocator::Heap { page, slot }) => {
                        // pool-first read: a committed row can live in a
                        // dirty buffer frame long before any flush
                        let data = crate::operator::scan::read_page_through_pool(
                            &self.ctx.buffer_pool,
                            &self.ctx.disk_manager,
                            page,
                        )
                        .await?;
                        pages_read += 1;
                        let Some(view) =
                            HeapPage::get_tuple_view_from_slice(&data, zyron_storage::SlotId(slot))
                        else {
                            continue;
                        };
                        if !crate::operator::doc_fetch::heap_row_visible(
                            &self.ctx,
                            self.heap_version,
                            view.header.xmin as u64,
                            view.header.xmax as u64,
                        ) {
                            continue;
                        }
                        decode_tuple_into_builders(
                            view.data,
                            &table_entry.columns,
                            &column_to_builder,
                            &mut builders,
                        );
                    }
                    Some(RowLocator::Columnar { file_id, sys_rowid }) => {
                        let Some(vals) = self.fetcher.columnar_row(file_id, sys_rowid) else {
                            continue;
                        };
                        for (b, v) in builders.iter_mut().zip(vals.iter()) {
                            b.push(v);
                        }
                    }
                    Some(RowLocator::Lake { file_id, ordinal }) => {
                        let Some(vals) = self.fetcher.lake_row(file_id, ordinal) else {
                            continue;
                        };
                        for (b, v) in builders.iter_mut().zip(vals.iter()) {
                            b.push(v);
                        }
                    }
                }
                scores.push(score);
                row_count += 1;
            }

            self.io_stats.record_batch(
                row_count as u64,
                pages_read * zyron_common::page::PAGE_SIZE as u64,
            );

            if row_count == 0 {
                self.finished = true;
                return Ok(None);
            }

            let mut batch = finalize_builders(builders);
            // Append relevance score as an additional Float64 column.
            batch.columns.push(crate::column::Column::new(
                crate::column::ColumnData::Float64(scores),
                zyron_common::TypeId::Float64,
            ));
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
