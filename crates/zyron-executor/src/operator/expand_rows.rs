//! Expands each input row into zero or more output rows.
//!
//! UNNEST walks arrays, FLATTEN walks a VARIANT document and UNPIVOT walks a
//! fixed list of column groups. All three have the same shape: read one value
//! per input row, decide how many output rows it yields, and write the
//! produced columns beside the input's own.
//!
//! The input's columns are never copied per produced row. A repeat vector
//! records, for each output row, which input row it came from, and the whole
//! carried column is gathered once with it. A ten-element array therefore
//! costs one offset walk and one gather, not ten row copies.

use std::rc::Rc;
use std::sync::Arc;

use zyron_common::{ArrayView, Result, TypeId, ZyronError};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::{ExpandSpec, LogicalColumn};

use crate::batch::{ColumnBuilder, DataBatch, decode_fixed_scalar, decode_varlen_scalar};
use crate::column::{Column, ColumnData, ScalarValue};
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Produces the output rows one expansion asks for.
pub struct ExpandRowsOperator {
    child: Box<dyn Operator>,
    spec: ExpandSpec,
    /// Input column positions that travel through, in output order
    carry: Vec<usize>,
    output_columns: Vec<LogicalColumn>,
    input_schema: Vec<LogicalColumn>,
    outer_input: bool,
    params: Vec<ScalarValue>,
    ctx: Arc<ExecutionContext>,
    batch_size: usize,
    /// Rows produced but not yet handed upward, held as one batch per
    /// produced column plus the repeat vector that addresses the input
    pending: Option<Pending>,
    finished: bool,
}

/// One input batch's expansion, ready to be sliced into output batches.
struct Pending {
    input: DataBatch,
    /// For each produced row, the input row it came from
    repeat: Vec<u32>,
    /// One builder's finished column per produced column, in output order
    produced: Vec<Column>,
    /// How many produced rows have already been handed upward
    emitted: usize,
}

impl ExpandRowsOperator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        child: Box<dyn Operator>,
        spec: ExpandSpec,
        carry: Vec<usize>,
        output_columns: Vec<LogicalColumn>,
        input_schema: Vec<LogicalColumn>,
        outer_input: bool,
        params: Vec<ScalarValue>,
        ctx: Arc<ExecutionContext>,
    ) -> Self {
        let batch_size = ctx.batch_size.max(1);
        Self {
            child,
            spec,
            carry,
            output_columns,
            input_schema,
            outer_input,
            params,
            ctx,
            batch_size,
            pending: None,
            finished: false,
        }
    }

    /// The produced columns' declared types, which are the trailing columns
    /// of the output after the carried ones.
    fn produced_types(&self) -> &[LogicalColumn] {
        &self.output_columns[self.carry.len()..]
    }

    /// Expands one input batch into a repeat vector and the produced columns.
    fn expand_batch(&self, input: DataBatch) -> Result<Pending> {
        let produced_types = self.produced_types();
        let mut repeat: Vec<u32> = Vec::with_capacity(input.num_rows * 4);
        let mut builders: Vec<ColumnBuilder> = produced_types
            .iter()
            .map(|c| ColumnBuilder::new(c.type_id, input.num_rows * 4))
            .collect();

        match &self.spec {
            ExpandSpec::Unnest {
                arrays,
                with_ordinality,
            } => {
                self.expand_unnest(&input, arrays, *with_ordinality, &mut repeat, &mut builders)?;
            }
            ExpandSpec::Flatten {
                document,
                path,
                recursive,
            } => {
                self.expand_flatten(
                    &input,
                    document,
                    path.as_deref(),
                    *recursive,
                    &mut repeat,
                    &mut builders,
                )?;
            }
            ExpandSpec::Unpivot {
                groups,
                labels,
                include_nulls,
            } => {
                self.expand_unpivot(
                    &input,
                    groups,
                    labels,
                    *include_nulls,
                    &mut repeat,
                    &mut builders,
                )?;
            }
        }

        Ok(Pending {
            input,
            repeat,
            produced: builders.into_iter().map(|b| b.finish()).collect(),
            emitted: 0,
        })
    }

    /// One column per array, zipped to the longest with the shorter padded
    /// with NULL, and a position column when WITH ORDINALITY was written.
    fn expand_unnest(
        &self,
        input: &DataBatch,
        arrays: &[BoundExpr],
        with_ordinality: bool,
        repeat: &mut Vec<u32>,
        builders: &mut [ColumnBuilder],
    ) -> Result<()> {
        // Each array expression is evaluated once over the whole batch, so
        // the per-row work is an offset walk over bytes already in hand
        let columns: Vec<Column> = arrays
            .iter()
            .map(|expr| evaluate(expr, input, &self.input_schema, &self.params))
            .collect::<Result<Vec<_>>>()?;
        // The type each produced column reports, which is what an element's
        // bytes are decoded to when the encoded value carries no type of its
        // own
        let declared_types: Vec<TypeId> = self
            .produced_types()
            .iter()
            .take(columns.len())
            .map(|c| c.type_id)
            .collect();

        // The encoded bytes of each column, borrowed rather than copied. A
        // column that is not binary backed holds no array, and every row of
        // it reads as absent
        let payloads: Vec<Option<&Vec<Vec<u8>>>> = columns
            .iter()
            .map(|c| match &c.data {
                ColumnData::Binary(v) => Some(v),
                _ => None,
            })
            .collect();
        // One view per array per row, parsed once and then indexed. Parsing
        // it per element would walk the header for every element of every
        // row, which is the whole cost of the operator
        let mut views: Vec<Option<ArrayView<'_>>> = vec![None; columns.len()];

        for row in 0..input.num_rows {
            // The row's length is the longest of its arrays, so a shorter
            // one pads with NULL rather than truncating the others
            let mut length = 0usize;
            for (i, column) in columns.iter().enumerate() {
                views[i] = match payloads[i] {
                    Some(bytes) if !column.is_null(row) => {
                        bytes.get(row).and_then(|b| ArrayView::parse(b))
                    }
                    _ => None,
                };
                if let Some(view) = &views[i] {
                    length = length.max(view.len());
                }
            }
            if length == 0 {
                if self.outer_input {
                    repeat.push(row as u32);
                    for builder in builders.iter_mut() {
                        builder.push(&ScalarValue::Null);
                    }
                }
                continue;
            }
            for element in 0..length {
                repeat.push(row as u32);
                for (i, view) in views.iter().enumerate() {
                    builders[i].push(&element_of(view.as_ref(), element, declared_types[i]));
                }
                if with_ordinality {
                    let ordinality = builders.len() - 1;
                    builders[ordinality].push(&ScalarValue::Int64(element as i64 + 1));
                }
            }
        }
        Ok(())
    }

    /// The six columns a document walk produces, one row per member reached.
    fn expand_flatten(
        &self,
        input: &DataBatch,
        document: &BoundExpr,
        path: Option<&str>,
        recursive: bool,
        repeat: &mut Vec<u32>,
        builders: &mut [ColumnBuilder],
    ) -> Result<()> {
        let documents = evaluate(document, input, &self.input_schema, &self.params)?;
        // The walk writes into the builders as it goes rather than into a
        // list of members first. A member's key, path and rendered value are
        // each one allocation, and staging them would allocate every one of
        // them twice
        let mut sink = FlattenSink {
            builders,
            repeat,
            row: 0,
            seq: 0,
        };
        for row in 0..input.num_rows {
            sink.row = row as u32;
            sink.seq = 0;
            if let Some(value) = parse_document(&documents, row)? {
                let root = match path {
                    None => Some(&value),
                    Some(p) => resolve_document_path(&value, p),
                };
                if let Some(root) = root {
                    walk_document(root, "", recursive, &mut sink);
                }
            }
            if sink.seq == 0 && self.outer_input {
                sink.emit_empty();
            }
        }
        Ok(())
    }

    /// One row per group, carrying the group's label and its values.
    fn expand_unpivot(
        &self,
        input: &DataBatch,
        groups: &[Vec<BoundExpr>],
        labels: &[BoundExpr],
        include_nulls: bool,
        repeat: &mut Vec<u32>,
        builders: &mut [ColumnBuilder],
    ) -> Result<()> {
        // Each group's value columns are evaluated once over the batch, so
        // a twelve-column unpivot is twelve column reads rather than twelve
        // reads per row
        let mut group_columns: Vec<Vec<Column>> = Vec::with_capacity(groups.len());
        for group in groups {
            let mut columns = Vec::with_capacity(group.len());
            for expr in group {
                columns.push(evaluate(expr, input, &self.input_schema, &self.params)?);
            }
            group_columns.push(columns);
        }
        let label_values: Vec<ScalarValue> = labels
            .iter()
            .map(|expr| match expr {
                BoundExpr::Literal { value, .. } => literal_scalar(value),
                _ => ScalarValue::Null,
            })
            .collect();

        for row in 0..input.num_rows {
            for (g, columns) in group_columns.iter().enumerate() {
                let all_null = columns.iter().all(|c| c.is_null(row));
                if all_null && !include_nulls {
                    continue;
                }
                repeat.push(row as u32);
                builders[0].push(&label_values[g]);
                for (slot, column) in columns.iter().enumerate() {
                    builders[slot + 1].push(&column.get_scalar(row));
                }
            }
        }
        Ok(())
    }

    /// Hands the next slice of a pending expansion upward, gathering the
    /// carried columns with the repeat vector for exactly that slice.
    fn take_pending(&mut self) -> Option<ExecutionBatch> {
        let pending = self.pending.as_mut()?;
        let total = pending.repeat.len();
        if pending.emitted >= total {
            self.pending = None;
            return None;
        }
        let start = pending.emitted;
        let end = (start + self.batch_size).min(total);
        pending.emitted = end;
        let slice = &pending.repeat[start..end];

        let mut columns: Vec<Column> = Vec::with_capacity(self.output_columns.len());
        for position in &self.carry {
            columns.push(pending.input.columns[*position].take(slice));
        }
        for column in &pending.produced {
            columns.push(column.slice(start, end - start));
        }
        if pending.emitted >= total {
            self.pending = None;
        }
        Some(ExecutionBatch::new(DataBatch::new(columns)))
    }
}

impl Operator for ExpandRowsOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                if let Some(batch) = self.take_pending() {
                    if batch.num_rows() > 0 {
                        return Ok(Some(batch));
                    }
                    continue;
                }
                if self.finished {
                    return Ok(None);
                }
                self.ctx.check_cancelled()?;
                match self.child.next().await? {
                    None => {
                        self.finished = true;
                        return Ok(None);
                    }
                    Some(input) => {
                        if input.batch.num_rows == 0 {
                            continue;
                        }
                        self.pending = Some(self.expand_batch(input.batch)?);
                    }
                }
            }
        })
    }
}

/// One element of an already parsed array, decoded to the declared output
/// type.
///
/// A row whose array is shorter than the longest in its group reads NULL,
/// which is the padding several zipped arrays ask for, and so does a row
/// with no array at all.
#[inline]
fn element_of(view: Option<&ArrayView<'_>>, element: usize, declared: TypeId) -> ScalarValue {
    let Some(view) = view else {
        return ScalarValue::Null;
    };
    // The element type the value was written with, which is what its bytes
    // are laid out as. The declared type is what the column reports
    let stored = view.element_type();
    match view.get(element) {
        Some(Some(payload)) => {
            let decode_as = if stored == TypeId::Null {
                declared
            } else {
                stored
            };
            if decode_as.fixed_size().unwrap_or(0) > 0 {
                decode_fixed_scalar(decode_as, payload)
            } else {
                decode_varlen_scalar(decode_as, payload)
            }
        }
        _ => ScalarValue::Null,
    }
}

/// The document text one row holds, or None when the row is null.
/// The document a row holds, parsed in place where the column allows it.
///
/// A text or binary column already holds the document's bytes, so the parse
/// reads them where they are. Going through `get_scalar` would copy the whole
/// document into a `ScalarValue` for every row before the parser saw a byte
/// of it, which for a million documents is a million copies of the input.
fn parse_document(column: &Column, row: usize) -> Result<Option<serde_json::Value>> {
    if column.is_null(row) {
        return Ok(None);
    }
    let parsed = match &column.data {
        ColumnData::Utf8(values) => match values.get(row) {
            Some(text) => serde_json::from_str(text),
            None => return Ok(None),
        },
        ColumnData::Binary(values) => match values.get(row) {
            Some(bytes) => serde_json::from_slice(bytes),
            None => return Ok(None),
        },
        _ => match column.get_scalar(row) {
            ScalarValue::Null => return Ok(None),
            other => serde_json::from_str(&format!("{other:?}")),
        },
    };
    match parsed {
        Ok(value) => Ok(Some(value)),
        Err(e) => Err(ZyronError::ExecutionError(format!(
            "FLATTEN read a value that is not a document, {e}"
        ))),
    }
}

/// Where a document walk writes the members it reaches.
///
/// The six produced columns are filled as the walk visits, so a member's
/// key, path and rendered value are allocated once and moved into their
/// column rather than staged in a list and copied out of it.
struct FlattenSink<'a> {
    builders: &'a mut [ColumnBuilder],
    repeat: &'a mut Vec<u32>,
    /// The input row the walk is on
    row: u32,
    /// How many members this row has produced, which is the seq column
    seq: i64,
}

impl FlattenSink<'_> {
    /// Records one member.
    fn emit(
        &mut self,
        key: Option<&str>,
        path: String,
        index: Option<usize>,
        value: String,
        parent: &Rc<str>,
    ) {
        self.repeat.push(self.row);
        self.seq += 1;
        self.builders[0].push(&ScalarValue::Int64(self.seq));
        match key {
            Some(k) => self.builders[1].push_owned(ScalarValue::Utf8(k.to_string())),
            None => self.builders[1].push_null(),
        }
        self.builders[2].push_owned(ScalarValue::Utf8(path));
        match index {
            Some(i) => self.builders[3].push(&ScalarValue::Int64(i as i64)),
            None => self.builders[3].push_null(),
        }
        self.builders[4].push_owned(ScalarValue::Utf8(value));
        self.builders[5].push_owned(ScalarValue::Utf8(parent.to_string()));
    }

    /// Records the one row an outer walk emits for an input that reached no
    /// member, carrying a null value.
    fn emit_empty(&mut self) {
        self.repeat.push(self.row);
        self.builders[0].push(&ScalarValue::Int64(0));
        for builder in self.builders.iter_mut().skip(1) {
            builder.push_null();
        }
    }
}

/// Walks a document, recording one member per object member and array
/// element. A recursive walk descends into nested containers depth first; a
/// shallow one reports the container's own members and stops.
fn walk_document(
    value: &serde_json::Value,
    prefix: &str,
    recursive: bool,
    sink: &mut FlattenSink<'_>,
) {
    walk_container(value, prefix, recursive, None, sink)
}

/// Walks one container, taking its own rendering from the caller when the
/// caller already made one.
///
/// A container that is descended into is rendered once, as the value of the
/// member that names it, and that rendering is handed down to serve as the
/// `this` column of every member inside it. Rendering it again on the way in
/// would render every interior node of a document twice.
fn walk_container(
    value: &serde_json::Value,
    prefix: &str,
    recursive: bool,
    rendered: Option<Rc<str>>,
    sink: &mut FlattenSink<'_>,
) {
    // A scalar has no members, so nothing is rendered for one
    let (map, items) = match value {
        serde_json::Value::Object(map) => (Some(map), None),
        serde_json::Value::Array(items) => (None, Some(items)),
        _ => return,
    };
    // One rendering of the container serves every member found in it.
    // Rendering it per member would render a twenty member object twenty
    // times over
    let parent: Rc<str> = match rendered {
        Some(text) => text,
        None => Rc::from(value.to_string().as_str()),
    };
    if let Some(map) = map {
        for (key, child) in map {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            emit_member(Some(key), path, None, child, recursive, &parent, sink);
        }
    }
    if let Some(items) = items {
        for (index, child) in items.iter().enumerate() {
            let path = format!("{prefix}[{index}]");
            emit_member(None, path, Some(index), child, recursive, &parent, sink);
        }
    }
}

/// Writes one member and descends into it when the walk is recursive.
///
/// The child's rendering is made once here and handed to the descent, which
/// is what keeps an interior container from being rendered a second time.
#[allow(clippy::too_many_arguments)]
fn emit_member(
    key: Option<&str>,
    path: String,
    index: Option<usize>,
    child: &serde_json::Value,
    recursive: bool,
    parent: &Rc<str>,
    sink: &mut FlattenSink<'_>,
) {
    let descend = recursive && (child.is_object() || child.is_array());
    if !descend {
        sink.emit(key, path, index, child.to_string(), parent);
        return;
    }
    let rendered: Rc<str> = Rc::from(child.to_string().as_str());
    // The path is the child's prefix, so it is kept for the descent and the
    // rendering travels with it rather than being made again
    sink.emit(key, path.clone(), index, rendered.to_string(), parent);
    walk_container(child, &path, recursive, Some(rendered), sink);
}

/// Follows a `path =>` argument to the value it names, or None when the
/// document has nothing there.
///
/// A segment is an object member name, `[n]` an array position, and `[*]`
/// the array itself, which is what makes `a.b[*]` start a walk over `b`'s
/// elements.
fn resolve_document_path<'a>(
    value: &'a serde_json::Value,
    path: &str,
) -> Option<&'a serde_json::Value> {
    let mut current = value;
    for raw in path.split('.') {
        let mut segment = raw;
        if let Some(open) = segment.find('[') {
            let (name, rest) = segment.split_at(open);
            if !name.is_empty() {
                current = current.get(name)?;
            }
            segment = rest;
            for part in segment.split_inclusive(']') {
                let inner = part.trim_start_matches('[').trim_end_matches(']');
                if inner == "*" {
                    // The array itself, whose elements the walk then reports
                    continue;
                }
                let index: usize = inner.parse().ok()?;
                current = current.get(index)?;
            }
        } else {
            current = current.get(segment)?;
        }
    }
    Some(current)
}

/// A bound literal as the value the operator writes.
fn literal_scalar(value: &zyron_parser::ast::LiteralValue) -> ScalarValue {
    use zyron_parser::ast::LiteralValue;
    match value {
        LiteralValue::String(s) => ScalarValue::Utf8(s.clone()),
        LiteralValue::Integer(i) => ScalarValue::Int64(*i),
        LiteralValue::Int128(i) => ScalarValue::Int128(*i),
        LiteralValue::Float(f) => ScalarValue::Float64(*f),
        LiteralValue::Decimal { digits, .. } => ScalarValue::Int128(*digits),
        LiteralValue::Boolean(b) => ScalarValue::Boolean(*b),
        LiteralValue::Null => ScalarValue::Null,
        LiteralValue::Interval(i) => ScalarValue::Interval(*i),
        LiteralValue::Bytes(b) => ScalarValue::Binary(b.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_reaches_a_nested_array() {
        let doc: serde_json::Value =
            serde_json::from_str(r#"{"a":{"b":[{"c":1},{"c":2}]}}"#).expect("parses");
        let reached = resolve_document_path(&doc, "a.b[*]").expect("resolves");
        assert!(reached.is_array());
        assert_eq!(reached.as_array().map(|a| a.len()), Some(2));

        let one = resolve_document_path(&doc, "a.b[1]").expect("resolves");
        assert_eq!(one.get("c").and_then(|v| v.as_i64()), Some(2));

        assert!(resolve_document_path(&doc, "a.missing").is_none());
    }

    /// Walks a document and reads back the six columns it produced.
    fn walk(json: &str, recursive: bool) -> Vec<Vec<ScalarValue>> {
        let doc: serde_json::Value = serde_json::from_str(json).expect("parses");
        let mut builders = vec![
            ColumnBuilder::new(TypeId::Int64, 8),
            ColumnBuilder::new(TypeId::Text, 8),
            ColumnBuilder::new(TypeId::Text, 8),
            ColumnBuilder::new(TypeId::Int64, 8),
            ColumnBuilder::new(TypeId::Variant, 8),
            ColumnBuilder::new(TypeId::Variant, 8),
        ];
        let mut repeat = Vec::new();
        {
            let mut sink = FlattenSink {
                builders: &mut builders,
                repeat: &mut repeat,
                row: 0,
                seq: 0,
            };
            walk_document(&doc, "", recursive, &mut sink);
        }
        let columns: Vec<Column> = builders.into_iter().map(|b| b.finish()).collect();
        (0..repeat.len())
            .map(|r| columns.iter().map(|c| c.get_scalar(r)).collect())
            .collect()
    }

    fn text_at(row: &[ScalarValue], column: usize) -> Option<String> {
        match &row[column] {
            ScalarValue::Utf8(s) => Some(s.clone()),
            _ => None,
        }
    }

    #[test]
    fn a_recursive_walk_reaches_a_leaf_three_levels_down() {
        let rows = walk(r#"{"a":{"b":{"c":7}}}"#, true);
        let paths: Vec<String> = rows
            .iter()
            .map(|r| text_at(r, 2).unwrap_or_default())
            .collect();
        assert_eq!(paths, vec!["a", "a.b", "a.b.c"]);
        let leaf = rows.last().expect("the leaf is reached");
        assert_eq!(text_at(leaf, 4).as_deref(), Some("7"));
        assert_eq!(text_at(leaf, 1).as_deref(), Some("c"));
        assert!(
            matches!(leaf[3], ScalarValue::Null),
            "an object member has no index"
        );
        // seq numbers the members of one input row from 1
        assert_eq!(rows[0][0], ScalarValue::Int64(1));
        assert_eq!(rows[2][0], ScalarValue::Int64(3));
    }

    #[test]
    fn a_shallow_walk_stops_at_the_first_level() {
        let rows = walk(r#"{"a":{"b":{"c":7}}}"#, false);
        assert_eq!(rows.len(), 1);
        assert_eq!(text_at(&rows[0], 2).as_deref(), Some("a"));
    }

    #[test]
    fn an_array_element_carries_its_index_and_no_key() {
        let rows = walk(r#"[10,20]"#, false);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1][3], ScalarValue::Int64(1));
        assert!(matches!(rows[1][1], ScalarValue::Null));
        assert_eq!(text_at(&rows[1], 2).as_deref(), Some("[1]"));
    }

    #[test]
    fn every_member_of_one_container_reads_the_same_parent() {
        let rows = walk(r#"{"a":1,"b":2}"#, false);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            text_at(&rows[0], 5),
            text_at(&rows[1], 5),
            "the container is rendered once and shared"
        );
    }

    #[test]
    fn a_scalar_document_reaches_no_member() {
        assert!(walk("7", true).is_empty());
        assert!(walk("\"text\"", true).is_empty());
    }
}
