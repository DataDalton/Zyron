//! Storage-format conversion for a whole table.
//!
//! Reads every row a table holds in one format and writes it in the other,
//! then leaves the catalog flip to the caller. The flip is the single point
//! at which the table changes format, so this module never touches the
//! catalog: a crash before the flip leaves work nobody can reach, and
//! startup reclaims it.
//!
//! Reading a heap table means both stores it can use: heap pages under the
//! statement's snapshot and, when the table has folded segments, the
//! columnar scan that applies the patch overlay and MVCC visibility. A
//! conversion that read only the heap would silently drop every folded row.

use std::sync::Arc;

use zyron_catalog::TableEntry;
use zyron_common::{PageId, Result, ZyronError};
use zyron_storage::HeapPage;

use crate::batch::{DataBatch, encode_scalar_value};
use crate::column::ScalarValue;
use crate::context::ExecutionContext;
use crate::operator::Operator;
use crate::operator::fk::decode_tuple_to_batch;
use zyron_planner::logical::LogicalColumn;

/// Every visible row of a heap-format table, column major, in the cell form
/// a lake data file stores.
pub async fn read_heap_rows(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
) -> Result<Vec<zyron_lake::ColumnData>> {
    let mut columns: Vec<zyron_lake::ColumnData> = table
        .columns
        .iter()
        .map(|c| zyron_lake::ColumnData {
            column_id: c.id.0 as u32,
            cells: Vec::new(),
        })
        .collect();

    let heap = ctx.get_heap_file(table.id).await?;
    let num_pages = heap.num_pages_cached() as u32;
    for page_num in 0..num_pages {
        ctx.check_cancelled()?;
        let page_id = PageId::new(table.heap_file_id, page_num as u64);
        let page_data = crate::operator::scan::read_page_through_pool(
            &ctx.buffer_pool,
            &ctx.disk_manager,
            page_id,
        )
        .await?;
        let header = HeapPage::heap_header_from_slice(&page_data);
        if header.slot_count == 0 {
            continue;
        }
        let page = HeapPage::from_bytes(page_data);
        for slot in 0..header.slot_count {
            let Some(view) = page.get_tuple_view(zyron_storage::SlotId(slot)) else {
                continue;
            };
            if view.is_deleted() || !view.header.is_visible_to(&ctx.snapshot) {
                continue;
            }
            let batch = decode_tuple_to_batch(view.data, table);
            append_batch_row(&mut columns, table, &batch, 0);
        }
    }

    // Folded rows are not in the heap. Reading them through the columnar
    // scan is what keeps a conversion from losing them
    if !table.columnar.segments.is_empty() {
        let logical: Vec<LogicalColumn> = table
            .columns
            .iter()
            .map(|c| LogicalColumn {
                table_idx: Some(0),
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                fractional_digits: c.fractional_digits,
            })
            .collect();
        let mut op = crate::operator::column_scan::ColumnScanOperator::new_for_dml(
            Arc::clone(ctx),
            table.id,
            logical,
            None,
        )?;
        while let Some(eb) = op.next().await? {
            for row in 0..eb.batch.num_rows {
                append_batch_row(&mut columns, table, &eb.batch, row);
            }
        }
    }

    Ok(columns)
}

/// Appends one batch row to the column-major cell buffers.
fn append_batch_row(
    columns: &mut [zyron_lake::ColumnData],
    table: &TableEntry,
    batch: &DataBatch,
    row: usize,
) {
    for (index, column) in table.columns.iter().enumerate() {
        let value_size = column.physical_type_id().fixed_size().unwrap_or(0);
        let scalar = batch.columns[index].get_scalar(row);
        columns[index].cells.push(match scalar {
            ScalarValue::Null => None,
            ref v => Some(encode_scalar_value(column.type_id, v, value_size)),
        });
    }
}

/// Turns column-major lake cells back into batches a heap insert accepts.
///
/// Batches are capped so a large table converts through bounded memory
/// rather than one allocation the size of the table.
pub fn cells_to_batches(
    table: &TableEntry,
    columns: &[zyron_lake::ColumnData],
    batch_rows: usize,
) -> Result<Vec<DataBatch>> {
    let row_count = columns.first().map(|c| c.cells.len()).unwrap_or(0);
    if row_count == 0 {
        return Ok(Vec::new());
    }
    if columns.len() != table.columns.len() {
        return Err(ZyronError::Internal(format!(
            "conversion produced {} columns for a table with {}",
            columns.len(),
            table.columns.len()
        )));
    }
    let logical: Vec<LogicalColumn> = table
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();

    let batch_rows = batch_rows.max(1);
    let mut out = Vec::with_capacity(row_count.div_ceil(batch_rows));
    let mut start = 0usize;
    while start < row_count {
        let end = (start + batch_rows).min(row_count);
        let mut builders = crate::batch::create_builders(&logical, end - start);
        for row in start..end {
            for (index, column) in table.columns.iter().enumerate() {
                let value_size = column.physical_type_id().fixed_size().unwrap_or(0);
                let scalar = match columns[index].cells[row].as_deref() {
                    None => ScalarValue::Null,
                    Some(cell) if value_size == 0 => {
                        crate::batch::decode_varlen_scalar(column.type_id, cell)
                    }
                    Some(cell) => crate::batch::decode_fixed_scalar(column.type_id, cell),
                };
                builders[index].push(&scalar);
            }
        }
        out.push(crate::batch::finalize_builders(builders));
        start = end;
    }
    Ok(out)
}
