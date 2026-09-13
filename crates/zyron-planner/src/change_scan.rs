//! Binding a read of a table's recorded changes.
//!
//! Two spellings reach the same node. `table_changes(t, from, to)` names a
//! table and a window directly. A change stream named in a FROM clause reads
//! from where it left off to the boundary its sources share, and additionally
//! records the advance its commit will make.
//!
//! Bounds are resolved here rather than in the operator, because EXPLAIN
//! reports the range a scan settled on and the number of change files it will
//! open, and a plan that carried `LATEST` unresolved could report neither

use zyron_catalog::{ChangeStreamEntry, ChangeStreamMode, ColumnId, TableEntry, TableId};
use zyron_common::{Result, TypeId, ZyronError};

use crate::ChangeFeedFacts;
use crate::binder::BoundExpr;
use crate::logical::{
    ChangeBound, ChangeMetadataColumn, ChangeScanSpec, ChangeScanWindow, ChangeStreamBinding,
    LogicalColumn,
};

/// The function name a change read is written as
pub const TABLE_CHANGES: &str = "table_changes";

/// Whether a name in a FROM clause opens a change read
pub fn is_change_function(name: &str) -> bool {
    name.eq_ignore_ascii_case(TABLE_CHANGES)
}

/// Reads a bound as the statement wrote it.
///
/// `LATEST` and `EARLIEST` are written as bare words, a whole number is a
/// commit version and a string is a timestamp, which is the same shape
/// `AS OF` takes
pub fn parse_bound(expr: &zyron_parser::ast::Expr, what: &str) -> Result<ChangeBound> {
    use zyron_parser::ast::{Expr, LiteralValue};
    match expr {
        Expr::Identifier(word) if word.eq_ignore_ascii_case("latest") => Ok(ChangeBound::Latest),
        Expr::Identifier(word) if word.eq_ignore_ascii_case("earliest") => {
            Ok(ChangeBound::Earliest)
        }
        Expr::Literal(LiteralValue::Integer(n)) if *n >= 0 => Ok(ChangeBound::Version(*n as u64)),
        Expr::Literal(LiteralValue::String(text)) => Ok(ChangeBound::Timestamp(
            zyron_common::interval::parse_timestamp_micros(text)?,
        )),
        other => Err(ZyronError::PlanError(format!(
            "{what} takes a version, a timestamp, LATEST or EARLIEST, found {other:?}"
        ))),
    }
}

/// Turns a written bound into the version that addresses it in this node's
/// own feed.
///
/// With no facts installed the bounds resolve to the widest window the
/// numbers allow, which is what an internal plan built before the server
/// finished starting sees, and the operator narrows it against the feed
fn resolve_bound(
    facts: Option<&dyn ChangeFeedFacts>,
    table_id: u32,
    bound: ChangeBound,
    is_start: bool,
) -> u64 {
    match bound {
        ChangeBound::Version(v) => v,
        ChangeBound::Timestamp(ts) => match facts {
            Some(facts) => facts.version_at_timestamp(table_id, ts),
            None => {
                if is_start {
                    0
                } else {
                    u64::MAX
                }
            }
        },
        ChangeBound::Earliest => match facts.and_then(|f| f.version_range(table_id)) {
            // The window is open at its lower end, so starting one below the
            // oldest change is what includes it
            Some((oldest, _)) => oldest.saturating_sub(1),
            None => 0,
        },
        ChangeBound::Latest => match facts.and_then(|f| f.version_range(table_id)) {
            Some((_, newest)) => newest,
            None => u64::MAX,
        },
    }
}

/// The message a read of a table with no feed carries
fn no_feed(name: &str) -> ZyronError {
    ZyronError::PlanError(format!(
        "table '{name}' has no change data feed, so it records no changes to read. Enable it \
         with ALTER TABLE {name} SET (change_data_feed = true)"
    ))
}

/// The message a read below what retention still holds carries
fn range_expired(name: &str, requested: u64, oldest: u64) -> ZyronError {
    ZyronError::PlanError(format!(
        "ChangeFeedRangeExpired: version {requested} of table '{name}' is older than the feed's \
         retention. The oldest change still held is version {oldest}, so a read from \
         {requested} would silently skip everything between them"
    ))
}

/// The data columns a change read produces, which are the source's own
pub fn data_columns(table: &TableEntry, table_idx: usize) -> Vec<LogicalColumn> {
    columns_of(table, table_idx, false)
}

/// The data columns a change read produces. With `as_of_change` every
/// column the table has ever had is included, dropped ones too, so a record
/// renders in the shape it was written under and a column dropped since
/// reads its recorded value rather than vanishing
pub fn columns_of(table: &TableEntry, table_idx: usize, as_of_change: bool) -> Vec<LogicalColumn> {
    table
        .columns
        .iter()
        .filter(|c| as_of_change || !c.dropped)
        .map(|c| LogicalColumn {
            table_idx: Some(table_idx),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            // Every data column of a change record reads NULL for a record
            // written before the column existed, so none of them is declared
            // non-null however the table declares it
            nullable: true,
            fractional_digits: c.fractional_digits,
        })
        .collect()
}

/// The metadata columns, as the binder registers them
pub fn metadata_columns(metadata: &[ChangeMetadataColumn], table_idx: usize) -> Vec<LogicalColumn> {
    metadata
        .iter()
        .map(|meta| LogicalColumn {
            table_idx: Some(table_idx),
            column_id: crate::logical::change_metadata_column_id(*meta),
            name: meta.name().to_string(),
            type_id: meta.type_id(),
            nullable: false,
            fractional_digits: None,
        })
        .collect()
}

/// Narrows the data columns to the ones a stream or a feed records.
///
/// A column outside the feed's `cdf_columns` list holds no value in any
/// record, so reading it would answer NULL for a value the table has. Naming
/// it is the honest answer, and that check runs in the caller. What this does
/// is drop the columns a stream's own COLUMNS list left out
pub fn narrow_columns(columns: Vec<LogicalColumn>, keep: &[ColumnId]) -> Vec<LogicalColumn> {
    if keep.is_empty() {
        return columns;
    }
    columns
        .into_iter()
        .filter(|c| keep.contains(&c.column_id))
        .collect()
}

/// Builds one source window, resolving its bounds and asking the facts what
/// the read will cost
#[allow(clippy::too_many_arguments)]
pub fn build_window(
    facts: Option<&dyn ChangeFeedFacts>,
    table: &TableEntry,
    from: ChangeBound,
    to: ChangeBound,
    from_timestamp: i64,
    to_timestamp: i64,
    change_types: Option<u8>,
) -> Result<ChangeScanWindow> {
    let table_id = table.id.0;
    if let Some(facts) = facts {
        if !facts.feed_enabled(table_id) {
            return Err(no_feed(&table.name));
        }
    } else if !table.cdf_enabled {
        return Err(no_feed(&table.name));
    }

    let from_exclusive = resolve_bound(facts, table_id, from, true);
    let mut to_inclusive = resolve_bound(facts, table_id, to, false);
    if to_inclusive < from_exclusive {
        to_inclusive = from_exclusive;
    }

    if let Some(facts) = facts {
        let floor = facts.purge_floor(table_id);
        // A window that starts at or below what a purge reclaimed would skip
        // the changes between them, which is exactly what a consumer must
        // never be handed without being told
        if floor > 0 && from_exclusive < floor {
            let oldest = facts
                .version_range(table_id)
                .map(|(oldest, _)| oldest)
                .unwrap_or(floor.saturating_add(1));
            return Err(range_expired(&table.name, from_exclusive, oldest));
        }
    }

    let (files_opened, files_pruned) = facts
        .map(|f| f.files_for_window(table_id, from_exclusive, to_inclusive))
        .unwrap_or((0, 0));
    let estimated_rows = facts
        .map(|f| f.rows_in_window(table_id, from_exclusive, to_inclusive))
        .unwrap_or(0);
    let consumed_to = facts.map(|_| estimated_rows).unwrap_or(0);

    Ok(ChangeScanWindow {
        table_id: table.id,
        table_name: table.name.clone(),
        branch: None,
        from_exclusive,
        to_inclusive,
        open_ended: to == ChangeBound::Latest,
        from_timestamp,
        to_timestamp,
        change_types,
        consumed_to,
        files_opened,
        files_pruned,
        estimated_rows,
    })
}

/// Refuses a read of a column the feed does not record
pub fn check_recorded_columns(
    facts: Option<&dyn ChangeFeedFacts>,
    table: &TableEntry,
    requested: &[LogicalColumn],
) -> Result<()> {
    let Some(facts) = facts else {
        return Ok(());
    };
    let recorded = facts.recorded_columns(table.id.0);
    if recorded.is_empty() {
        return Ok(());
    }
    for column in requested {
        if !recorded.contains(&column.column_id.0) {
            return Err(ZyronError::PlanError(format!(
                "column '{}' is outside the change data feed's cdf_columns list on table '{}', \
                 so the feed holds no value for it. Add it to cdf_columns or leave it out of \
                 the query",
                column.name, table.name
            )));
        }
    }
    Ok(())
}

/// Refuses a read of `update_preimage` on a feed that keeps one image
pub fn check_before_image(
    facts: Option<&dyn ChangeFeedFacts>,
    table: &TableEntry,
    change_types: Option<u8>,
) -> Result<()> {
    let Some(facts) = facts else {
        return Ok(());
    };
    let asks_for_preimage = change_types
        .is_some_and(|mask| mask & (1 << zyron_common::CHANGE_TYPE_UPDATE_PREIMAGE) != 0);
    if asks_for_preimage && !facts.before_image(table.id.0) {
        return Err(ZyronError::PlanError(format!(
            "table '{}' records its change data feed without before images, so it holds no \
             update_preimage rows. Turn them on with ALTER TABLE {} SET (cdf_before_image = \
             true), which applies to changes recorded from then on",
            table.name, table.name
        )));
    }
    Ok(())
}

/// The change kinds a stream in this mode admits
pub fn mode_change_types(mode: ChangeStreamMode) -> Option<u8> {
    match mode {
        ChangeStreamMode::Standard => None,
        // An append-only stream skips updates and deletes at read time, which
        // is cheaper than reading and discarding them
        ChangeStreamMode::AppendOnly => Some(1 << zyron_common::CHANGE_TYPE_INSERT),
    }
}

/// How a stream is read, from the options written after it in a FROM clause
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamReadOptions {
    /// Reads without taking the position lock and without advancing
    pub peek: bool,
    /// Renders each record through the layout it was written under
    pub as_of_change: bool,
    /// The most records one read takes past the position, per source
    pub max_rows: Option<u64>,
}

/// Builds the spec for a read of one change stream
pub fn stream_spec(
    facts: Option<&dyn ChangeFeedFacts>,
    entry: &ChangeStreamEntry,
    tables: &[std::sync::Arc<TableEntry>],
    table_idx: usize,
    options: StreamReadOptions,
    predicate: Option<BoundExpr>,
) -> Result<ChangeScanSpec> {
    let StreamReadOptions {
        peek,
        as_of_change,
        max_rows,
    } = options;
    if entry.stale {
        return Err(ZyronError::PlanError(format!(
            "ChangeStreamStale: change stream '{}' cannot be read, {}. Move it with \
             ALTER CHANGE STREAM {} RESET TO VERSION <n>",
            entry.name, entry.stale_reason, entry.name
        )));
    }
    if entry.needs_attention {
        return Err(ZyronError::PlanError(format!(
            "change stream '{}' needs attention and yields nothing until it is corrected: {}",
            entry.name, entry.attention_reason
        )));
    }

    let change_types = mode_change_types(entry.mode);
    let multi = entry.source.is_multi_table();

    // The boundary every source has reached, so a transaction that touched
    // two of them is either wholly inside the window or wholly outside it
    let shared = if multi {
        tables
            .iter()
            .filter_map(|table| facts.and_then(|f| f.version_range(table.id.0)))
            .map(|(_, newest)| newest)
            .filter(|newest| *newest > 0)
            .min()
    } else {
        None
    };

    let mut windows = Vec::with_capacity(tables.len());
    for table in tables {
        let position = entry.position_of(table.id.0);
        let latest = facts
            .and_then(|f| f.version_range(table.id.0))
            .map(|(_, newest)| newest)
            .unwrap_or(u64::MAX);
        let to = match shared {
            Some(shared) => latest.min(shared),
            None => latest,
        };
        let mut window = build_window(
            facts,
            table,
            ChangeBound::Version(position),
            ChangeBound::Version(to.max(position)),
            i64::MIN,
            i64::MAX,
            change_types,
        )?;
        // A stream's position replicates as the count of records it has
        // consumed, and the window's upper end is the place that count names
        window.consumed_to = facts
            .map(|f| f.rows_in_window(table.id.0, 0, window.to_inclusive))
            .unwrap_or(0);
        // A stream created on a branch reads the branch's feed, whatever
        // branch the session reading it is on
        window.branch = entry.branch;
        windows.push(window);
    }

    let source_table = tables.first().ok_or_else(|| {
        ZyronError::PlanError(format!(
            "change stream '{}' names no source table",
            entry.name
        ))
    })?;
    // A column the feed does not record is refused when the plan settles on
    // reading it, after projection pushdown has dropped the ones nothing
    // asked for, so a query over the recorded columns of a narrowed feed
    // runs and one over an unrecorded column is named
    let all_columns = if multi {
        multi_table_columns(tables, table_idx)?
    } else {
        columns_of(source_table, table_idx, as_of_change)
    };
    let mut columns = all_columns.clone();
    if let Some(keep) = &entry.columns {
        columns = narrow_columns(columns, keep);
    }
    // The stream's stored WHERE reads the source's columns whether or not
    // its COLUMNS list exposes them, so the ones it names and the list left
    // out are decoded for the predicate alone
    let filter_columns = match &predicate {
        Some(predicate) => filter_only_columns(predicate, &columns, &all_columns, table_idx),
        None => Vec::new(),
    };

    let metadata = if multi {
        ChangeMetadataColumn::multi_table().to_vec()
    } else {
        ChangeMetadataColumn::single_table().to_vec()
    };

    // A stream created with SHOW INITIAL ROWS yields the source's existing
    // rows on its first read, and the entry says whether that read has
    // committed yet
    let initial_rows = entry.initial_rows_pending;

    Ok(ChangeScanSpec {
        windows,
        stream: Some(ChangeStreamBinding {
            stream_id: entry.id,
            stream_name: entry.name.clone(),
            peek,
            max_rows,
        }),
        as_of_change,
        initial_rows,
        data_columns: columns,
        filter_columns,
        metadata,
        predicate,
        table_idx,
    })
}

/// The source columns a predicate reads that the scan does not output.
///
/// Each is decoded beside the output columns so the predicate has its
/// value, and dropped before the row leaves the scan. A metadata column is
/// never here, every scan carries all of them
pub fn filter_only_columns(
    predicate: &BoundExpr,
    output: &[LogicalColumn],
    source: &[LogicalColumn],
    table_idx: usize,
) -> Vec<LogicalColumn> {
    let mut extra: Vec<LogicalColumn> = Vec::new();
    for reference in crate::optimizer::rules::predicate_pushdown::collect_column_refs(predicate) {
        if reference.table_idx != table_idx
            || reference.column_id.0 >= crate::logical::CHANGE_METADATA_COLUMN_BASE
            || output.iter().any(|c| c.column_id == reference.column_id)
            || extra.iter().any(|c| c.column_id == reference.column_id)
        {
            continue;
        }
        if let Some(column) = source.iter().find(|c| c.column_id == reference.column_id) {
            extra.push(column.clone());
        }
    }
    extra
}

/// The columns a stream over several tables yields, every column any of
/// them has, by name, in the order the tables list them, addressed by the
/// ids of the first table that has each.
///
/// A source that lacks one of them yields NULL for it. A name two sources
/// hold at different types is refused, because one column cannot carry both
pub fn multi_table_columns(
    tables: &[std::sync::Arc<TableEntry>],
    table_idx: usize,
) -> Result<Vec<LogicalColumn>> {
    let mut out: Vec<LogicalColumn> = Vec::new();
    let mut owner: Vec<&TableEntry> = Vec::new();
    for table in tables {
        for column in data_columns(table, table_idx) {
            match out.iter().position(|c| c.name == column.name) {
                Some(at) => {
                    if out[at].type_id != column.type_id {
                        return Err(ZyronError::PlanError(format!(
                            "column '{}' is {:?} on table '{}' and {:?} on table '{}', so a \
                             change stream over both cannot yield it as one column",
                            column.name,
                            out[at].type_id,
                            owner[at].name,
                            column.type_id,
                            table.name
                        )));
                    }
                }
                None => {
                    out.push(column);
                    owner.push(table);
                }
            }
        }
    }
    Ok(out)
}

/// The scan's columns as one source table holds them, resolved by name.
///
/// A scan over several tables addresses its columns by the first table's
/// ids, and each other table's changes decode through its own ids for the
/// same names. None for a name the table does not have, whose value that
/// source yields as NULL
pub fn columns_on(table: &TableEntry, columns: &[LogicalColumn]) -> Vec<Option<LogicalColumn>> {
    columns
        .iter()
        .map(|column| {
            // A column the scan addresses by the table's own id is that
            // column whether or not it has since been dropped. One addressed
            // by another source's id is found by name among the live ones
            table
                .columns
                .iter()
                .find(|c| c.id == column.column_id && c.name == column.name)
                .or_else(|| table.live_columns().find(|c| c.name == column.name))
                .map(|own| LogicalColumn {
                    column_id: own.id,
                    ..column.clone()
                })
        })
        .collect()
}

/// The type a metadata column's literal is compared at, for a predicate the
/// scan pushes into its window bounds
pub fn metadata_type(column: ChangeMetadataColumn) -> TypeId {
    column.type_id()
}

/// The table a `table_changes` call names, as written
pub fn table_argument(args: &[zyron_parser::ast::FunctionArg]) -> Result<String> {
    use zyron_parser::ast::{Expr, FunctionArg, LiteralValue};
    let first = args.first().ok_or_else(|| {
        ZyronError::PlanError(
            "table_changes(<table>, <from>, <to>) needs the table to read as its first argument"
                .to_string(),
        )
    })?;
    match first {
        FunctionArg::Unnamed(Expr::Identifier(name)) => Ok(name.clone()),
        FunctionArg::Unnamed(Expr::QualifiedIdentifier { table, column }) => {
            Ok(format!("{table}.{column}"))
        }
        FunctionArg::Unnamed(Expr::Literal(LiteralValue::String(name))) => Ok(name.clone()),
        FunctionArg::Named { name, .. } => Err(ZyronError::PlanError(format!(
            "table_changes takes the table to read as its first argument, and '{name}' was \
             written as a named one"
        ))),
        other => Err(ZyronError::PlanError(format!(
            "table_changes takes a table name as its first argument, found {other:?}"
        ))),
    }
}

/// The window bounds a `table_changes` call carries, positional or named.
///
/// `table_changes(t, 5, 10)` and `table_changes(t, start_version => 5,
/// end_version => 10)` are the same call. `to` left out reads to LATEST
pub fn bound_arguments(
    args: &[zyron_parser::ast::FunctionArg],
) -> Result<(ChangeBound, ChangeBound, bool)> {
    use zyron_parser::ast::FunctionArg;
    let mut from = None;
    let mut to = None;
    let mut as_of_change = false;
    let mut positional = Vec::new();
    for arg in args.iter().skip(1) {
        match arg {
            FunctionArg::Unnamed(expr) => positional.push(expr),
            FunctionArg::Named { name, value } => {
                let lowered = name.to_ascii_lowercase();
                match lowered.as_str() {
                    "schema" => as_of_change = schema_is_as_of_change(value)?,
                    "start_version" | "from_version" | "start" | "from" => {
                        from = Some(parse_bound(value, "start_version")?)
                    }
                    "end_version" | "to_version" | "end" | "to" => {
                        to = Some(parse_bound(value, "end_version")?)
                    }
                    "start_timestamp" | "from_timestamp" => {
                        from = Some(parse_bound(value, "start_timestamp")?)
                    }
                    "end_timestamp" | "to_timestamp" => {
                        to = Some(parse_bound(value, "end_timestamp")?)
                    }
                    other => {
                        return Err(ZyronError::PlanError(format!(
                            "table_changes does not take an argument called '{other}'. It reads \
                             start_version, end_version, start_timestamp, end_timestamp and \
                             schema"
                        )));
                    }
                }
            }
            FunctionArg::Wildcard => {
                return Err(ZyronError::PlanError(
                    "table_changes does not accept `*`".to_string(),
                ));
            }
        }
    }
    if let Some(expr) = positional.first() {
        from = Some(parse_bound(expr, "the start of the range")?);
    }
    if let Some(expr) = positional.get(1) {
        to = Some(parse_bound(expr, "the end of the range")?);
    }
    if positional.len() > 2 {
        return Err(ZyronError::PlanError(
            "table_changes(<table>, <from>, <to>) takes at most three positional arguments"
                .to_string(),
        ));
    }
    // A start left out reads from the oldest change the feed still holds,
    // which is what a first read of a table's history asks for. Reading to
    // the newest change is what a consumer polling for what is new asks
    // for, so it is what leaving the end out means
    Ok((
        from.unwrap_or(ChangeBound::Earliest),
        to.unwrap_or(ChangeBound::Latest),
        as_of_change,
    ))
}

/// Reads the `schema` argument of `table_changes`, `'current'` or
/// `'as_of_change'`
fn schema_is_as_of_change(value: &zyron_parser::ast::Expr) -> Result<bool> {
    use zyron_parser::ast::{Expr, LiteralValue};
    let word = match value {
        Expr::Literal(LiteralValue::String(word)) => word.clone(),
        Expr::Identifier(word) => word.clone(),
        other => {
            return Err(ZyronError::PlanError(format!(
                "schema takes 'current' or 'as_of_change', found {other:?}"
            )));
        }
    };
    match word.to_ascii_lowercase().as_str() {
        "current" => Ok(false),
        "as_of_change" => Ok(true),
        other => Err(ZyronError::PlanError(format!(
            "schema takes 'current' or 'as_of_change', not '{other}'"
        ))),
    }
}

/// Reads the options a relation carried, for a change read.
///
/// `peek` reads a stream without taking its position lock and without
/// advancing it. `schema` decides whether a record renders through the
/// table's current columns or the ones it was written under. `max_rows`
/// bounds how many records one read takes past the position
pub fn read_options(options: &[zyron_parser::ast::TableOption]) -> Result<StreamReadOptions> {
    use zyron_parser::ast::TableOptionValue as V;
    let mut peek = false;
    let mut as_of_change = false;
    let mut max_rows = None;
    for option in options {
        match option.key.to_ascii_lowercase().as_str() {
            "max_rows" => {
                max_rows = match &option.value {
                    V::Integer(n) if *n > 0 => Some(*n as u64),
                    V::String(word) => match word.parse::<u64>() {
                        Ok(n) if n > 0 => Some(n),
                        _ => {
                            return Err(ZyronError::PlanError(format!(
                                "max_rows takes a whole number above zero, found '{word}'"
                            )));
                        }
                    },
                    other => {
                        return Err(ZyronError::PlanError(format!(
                            "max_rows takes a whole number above zero, found {other:?}"
                        )));
                    }
                };
            }
            "peek" => {
                peek = match &option.value {
                    V::Boolean(b) => *b,
                    V::Identifier(word) => word.eq_ignore_ascii_case("true"),
                    V::String(word) => word.eq_ignore_ascii_case("true"),
                    other => {
                        return Err(ZyronError::PlanError(format!(
                            "peek takes true or false, found {other:?}"
                        )));
                    }
                };
            }
            "schema" => {
                let word = match &option.value {
                    V::String(word) => word.clone(),
                    V::Identifier(word) => word.clone(),
                    other => {
                        return Err(ZyronError::PlanError(format!(
                            "schema takes 'current' or 'as_of_change', found {other:?}"
                        )));
                    }
                };
                as_of_change = match word.to_ascii_lowercase().as_str() {
                    "current" => false,
                    "as_of_change" => true,
                    other => {
                        return Err(ZyronError::PlanError(format!(
                            "schema takes 'current' or 'as_of_change', not '{other}'"
                        )));
                    }
                };
            }
            other => {
                return Err(ZyronError::PlanError(format!(
                    "a relation in a FROM clause does not read an option called '{other}'. A \
                     change stream reads peek, schema and max_rows"
                )));
            }
        }
    }
    Ok(StreamReadOptions {
        peek,
        as_of_change,
        max_rows,
    })
}

/// Builds the spec for a `table_changes` call
#[allow(clippy::too_many_arguments)]
pub fn table_function_spec(
    facts: Option<&dyn ChangeFeedFacts>,
    table: &TableEntry,
    table_idx: usize,
    from: ChangeBound,
    to: ChangeBound,
    as_of_change: bool,
) -> Result<ChangeScanSpec> {
    let window = build_window(facts, table, from, to, i64::MIN, i64::MAX, None)?;
    let columns = columns_of(table, table_idx, as_of_change);
    Ok(ChangeScanSpec {
        windows: vec![window],
        stream: None,
        as_of_change,
        initial_rows: false,
        data_columns: columns,
        filter_columns: Vec::new(),
        metadata: ChangeMetadataColumn::single_table().to_vec(),
        predicate: None,
        table_idx,
    })
}

/// The table a change scan reads when it names exactly one
pub fn sole_table(spec: &ChangeScanSpec) -> Option<TableId> {
    match spec.windows.as_slice() {
        [window] => Some(window.table_id),
        _ => None,
    }
}
