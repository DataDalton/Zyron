//! Values for the columns a table's clustering expressions are stored in.
//!
//! An expression cluster key is stored in a column of its own so file
//! statistics cover it and pruning treats it as a column. That only works if
//! every write fills it, so this is where a lake insert computes those
//! values.
//!
//! Evaluation is per batch, not per row. `expr::evaluate` takes a whole
//! `DataBatch` and returns a whole `Column` through the compute kernels, so
//! one call covers every row a statement writes. A row loop here would cost
//! one bound-tree walk per row, which is the difference the batch interface
//! exists to remove.
//!
//! Binding is cached by the expression's canonical hash. The hash is the
//! identity of the canonical form, so a changed expression is a different
//! key and the cache needs no invalidation: an entry can only ever be
//! correct for the expression that produced its key.

use std::sync::Arc;

use scc::HashMap as SccHashMap;
use zyron_catalog::TableEntry;
use zyron_catalog::schema::DerivedColumnEntry;
use zyron_common::{Result, ZyronError};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::Column;

/// Bound expressions by table and canonical hash.
///
/// Parsing and binding an expression costs more than evaluating it over a
/// batch, so a table taking many small writes would spend most of an insert
/// re-deriving something that cannot have changed. The table id is part of
/// the key because a canonical form addresses columns by id, and the same
/// ids mean different columns in different tables
static BOUND_EXPRESSIONS: std::sync::OnceLock<SccHashMap<(u32, u64), Arc<BoundExpr>>> =
    std::sync::OnceLock::new();

fn cache() -> &'static SccHashMap<(u32, u64), Arc<BoundExpr>> {
    BOUND_EXPRESSIONS.get_or_init(SccHashMap::new)
}

/// Parses and binds one clustering expression against its table.
///
/// Goes through the same binder the planner uses, so the expression a query
/// writes and the one a write recomputes resolve and type identically. A
/// divergence there would store values the query's predicate does not match
pub async fn bind_derived(
    catalog: &zyron_catalog::Catalog,
    table: &Arc<TableEntry>,
    derived: &DerivedColumnEntry,
) -> Result<Arc<BoundExpr>> {
    let key = (table.id.0, derived.canonical_hash);
    if let Some(found) = cache().read_sync(&key, |_, v| Arc::clone(v)) {
        return Ok(found);
    }

    let bound = Arc::new(bind_expression_text(catalog, table, &derived.sql).await?);
    let _ = cache().insert_sync(key, Arc::clone(&bound));
    Ok(bound)
}

/// Parses and binds one expression's stored text against its table.
///
/// The write path reaches this on every insert into a clustered table, and
/// the DDL that declares the expression reaches it once to prove the text
/// it is about to store can be read back. Both go through the binder a
/// query uses, so the expression a query writes and the one a write
/// recomputes resolve and type identically
pub async fn bind_expression_text(
    catalog: &zyron_catalog::Catalog,
    table: &Arc<TableEntry>,
    sql: &str,
) -> Result<BoundExpr> {
    let statement = format!("SELECT {sql} FROM t");
    let parsed = zyron_parser::parse(&statement).map_err(|e| {
        ZyronError::ExecutionError(format!(
            "clustering expression \"{sql}\" does not parse: {e}"
        ))
    })?;
    let expr = match parsed.into_iter().next() {
        Some(zyron_parser::Statement::Select(select)) => select
            .projections
            .into_iter()
            .next()
            .and_then(|item| match item {
                zyron_parser::ast::SelectItem::Expr(expr, _) => Some(expr),
                _ => None,
            })
            .ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "clustering expression \"{sql}\" projected nothing"
                ))
            })?,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "clustering expression \"{sql}\" is not an expression"
            )));
        }
    };

    let schema_entry = catalog.get_schema_by_id(table.schema_id)?;
    let resolver = catalog.resolver(schema_entry.database_id, vec![schema_entry.name.clone()]);
    let mut binder = zyron_planner::Binder::new(resolver, catalog);
    binder.bind_scalar_over_table(table, &expr).await
}

/// Evaluates every clustering expression over one batch, in the column order
/// the lake schema holds them.
///
/// Returns one `Column` per derived column, each covering the whole batch.
/// The caller appends their cells beside the stored columns, so the writer
/// sees a batch that covers the schema and computes statistics over the
/// expression exactly as it does over a stored column
pub async fn evaluate_derived(
    catalog: &zyron_catalog::Catalog,
    table: &Arc<TableEntry>,
    batch: &DataBatch,
    schema: &[LogicalColumn],
) -> Result<Vec<(u32, Column)>> {
    let mut out = Vec::with_capacity(table.cluster.derived.len());
    for derived in &table.cluster.derived {
        let bound = bind_derived(catalog, table, derived).await?;
        let column = crate::expr::evaluate(&bound, batch, schema, &[]).map_err(|e| {
            ZyronError::ExecutionError(format!(
                "clustering expression \"{}\" could not be computed for this write: {e}",
                derived.sql
            ))
        })?;
        out.push((derived.column_id, column));
    }
    Ok(out)
}

/// Computes the stored values for every clustering expression across one
/// statement's batches, encoded the way the lake writer takes them.
///
/// One evaluation per batch per expression, so a statement writing a million
/// rows in a handful of batches walks the bound tree a handful of times
pub async fn derived_column_data(
    catalog: &zyron_catalog::Catalog,
    table: &Arc<TableEntry>,
    batches: &[DataBatch],
) -> Result<Vec<zyron_lake::ColumnData>> {
    if table.cluster.derived.is_empty() {
        return Ok(Vec::new());
    }
    let schema = logical_schema(table);
    let mut out: Vec<zyron_lake::ColumnData> = table
        .cluster
        .derived
        .iter()
        .map(|d| zyron_lake::ColumnData::with_capacity(d.column_id, 0, 0))
        .collect();
    // One buffer refilled per cell rather than one allocation per cell
    let mut scratch: Vec<u8> = Vec::new();
    for batch in batches {
        let computed = evaluate_derived(catalog, table, batch, &schema).await?;
        for (slot, (_, column)) in out.iter_mut().zip(computed.into_iter()) {
            let type_id = column.type_id;
            let value_size = type_id.fixed_size().unwrap_or(0);
            for r in 0..batch.num_rows {
                match column.get_scalar(r) {
                    crate::column::ScalarValue::Null => slot.push(None),
                    ref v => {
                        scratch.clear();
                        crate::batch::encode_scalar_value_into(
                            &mut scratch,
                            type_id,
                            v,
                            value_size,
                        );
                        slot.push(Some(&scratch));
                    }
                }
            }
        }
    }
    Ok(out)
}

/// One expression proved storable, with what a caller needs to register it
pub struct StorableExpression {
    /// The identity, the text that will be stored, and the columns it reads
    pub canonical: zyron_planner::cluster_expr::CanonicalExpr,
    /// The expression as it reads back, which is the one every write
    /// evaluates and the one its result type has to come from
    pub bound: BoundExpr,
}

/// Proves an expression can be stored, read back, and computed, or refuses
/// it where it is declared.
///
/// Every statement that persists an expression has to pass all of this, and
/// the steps only mean anything together, so they live in one function
/// rather than in each caller. A caller that ran three of the four would
/// create a table whose every later write fails, and the statement that
/// created it would have reported success.
///
/// What is proved, and what each step is protecting against:
///
/// 1. It canonicalizes. Anything volatile, aggregate, subquery-bearing or
///    constant has no per-row value a column could hold.
/// 2. Its rendering re-parses and re-binds. The write path recomputes the
///    column by reading the stored text back, so the renderer has to be the
///    inverse of the parser. This is proved rather than assumed because a
///    renderer that is wrong for one node shape fails at insert time, long
///    after the statement that introduced it.
/// 3. What it reads back canonicalizes to the same identity. Text that
///    parses but means something else is worse than text that does not
///    parse, because nothing would notice.
/// 4. The evaluator can compute it. The binder knowing a function does not
///    mean a kernel implements it.
///
/// Everything after step two is proved about the expression **as it reads
/// back**, because that is the one the write path will evaluate.
///
/// `subject` names the thing being declared, for the refusals: something
/// like `derived column "yr"` or `CLUSTER BY expression on events`
pub async fn prove_storable(
    catalog: &zyron_catalog::Catalog,
    table: &Arc<TableEntry>,
    expr: &zyron_parser::ast::Expr,
    subject: &str,
) -> Result<StorableExpression> {
    let schema_entry = catalog.get_schema_by_id(table.schema_id)?;
    let resolver = catalog.resolver(schema_entry.database_id, vec![schema_entry.name.clone()]);
    let mut binder = zyron_planner::Binder::new(resolver, catalog);
    let declared = binder.bind_scalar_over_table(table, expr).await?;

    let canonical = zyron_planner::cluster_expr::canonicalize(&declared, &table.columns)
        .ok_or_else(|| {
            ZyronError::ParseError(format!(
                "{subject} is not storable: the expression must be deterministic, read at \
                 least one column of the table, and not be an aggregate, a subquery or a \
                 bare column"
            ))
        })?;

    let bound = bind_expression_text(catalog, table, &canonical.sql)
        .await
        .map_err(|e| {
            ZyronError::ParseError(format!(
                "{subject} is not storable, because it does not read back as written: {e}"
            ))
        })?;
    if zyron_planner::cluster_expr::canonicalize(&bound, &table.columns).map(|c| c.canonical_hash)
        != Some(canonical.canonical_hash)
    {
        return Err(ZyronError::ParseError(format!(
            "{subject} is not storable: \"{}\" reads back as a different expression",
            canonical.sql
        )));
    }

    let logical = logical_schema(table);
    check_evaluable(table, &bound, &logical)
        .map_err(|e| ZyronError::ParseError(format!("{subject} cannot be computed: {e}")))?;

    // Legal and expensive, so it is said while the operator can still pick
    // another expression
    if let Some(function) = row_at_a_time_function(&bound) {
        tracing::warn!(
            table = %table.name,
            subject = %subject,
            function = %function,
            expression = %canonical.sql,
            "expression is evaluated a row at a time, so every write into this table pays \
             per-row cost for it"
        );
    }

    Ok(StorableExpression { canonical, bound })
}

/// Functions the evaluator answers a row at a time.
///
/// Everything else reaches a compute kernel that runs over the whole column,
/// so an expression built only from those costs one pass per batch. One of
/// these turns a write into a per-row walk instead, which is worth telling an
/// operator about while they can still choose a different expression
const ROW_AT_A_TIME_FUNCTIONS: &[&str] = &[
    "match_against",
    "vector_distance_cosine",
    "vector_distance_l2",
    "vector_distance_dot",
    "row_diff",
];

/// The first row-at-a-time function an expression uses, if any.
///
/// Reported at declaration time rather than discovered from a slow insert
/// later
pub fn row_at_a_time_function(expr: &BoundExpr) -> Option<String> {
    match expr {
        BoundExpr::Nested(inner) => row_at_a_time_function(inner),
        BoundExpr::Function { name, args, .. } => {
            let lowered = name.to_ascii_lowercase();
            if ROW_AT_A_TIME_FUNCTIONS.contains(&lowered.as_str()) {
                return Some(lowered);
            }
            args.iter().find_map(row_at_a_time_function)
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            row_at_a_time_function(left).or_else(|| row_at_a_time_function(right))
        }
        BoundExpr::UnaryOp { expr, .. } => row_at_a_time_function(expr),
        BoundExpr::Cast { expr, .. } => row_at_a_time_function(expr),
        _ => None,
    }
}

/// Proves an expression can actually be computed, by evaluating it over one
/// probe row built from its source columns.
///
/// Declaration is the place to find out. Without this a CREATE TABLE naming
/// a function the engine does not implement is accepted, and every insert
/// afterwards fails on a table the statement said was created
pub fn check_evaluable(
    table: &Arc<TableEntry>,
    bound: &BoundExpr,
    schema: &[LogicalColumn],
) -> Result<()> {
    // One all-null row rather than none. A literal broadcasts to the batch
    // length, and a function that reads a constant argument out of its first
    // element needs that element to exist, so a zero-row probe would report
    // an expression as uncomputable that computes perfectly well
    let columns: Vec<Column> = table
        .columns
        .iter()
        .map(|c| Column::null_column(c.type_id, 1))
        .collect();
    let batch = DataBatch {
        columns,
        num_rows: 1,
    };
    crate::expr::evaluate(bound, &batch, schema, &[]).map(|_| ())
}

/// The logical schema an expression binds against, one entry per table
/// column in catalog order
pub fn logical_schema(table: &TableEntry) -> Vec<LogicalColumn> {
    table
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
        .collect()
}
