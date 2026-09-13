//! Lowering from a bound SQL predicate to the lake's typed predicate IR.
//!
//! The lowering is exact or absent. A returned predicate selects exactly
//! the rows the bound expression selects, including SQL three-valued
//! semantics over NULL, so it is equally sound for skipping files during a
//! scan and for recording a predicate delete. Anything that cannot be
//! proven equivalent lowers to None, and the caller falls back: a scan
//! keeps every file, a delete refuses rather than deleting the wrong rows.
//!
//! Literal conversion is by the column's storage representation, using the
//! same parse the executor uses for the same comparison, so a lowered
//! constant compares byte-identically to the stored cell.

use zyron_catalog::schema::{ColumnEntry, DerivedColumnEntry};
use zyron_common::TypeId;
use zyron_lake::{CompareOp, LakePredicate, LakeValue};
use zyron_parser::ast::{BinaryOperator, LiteralValue, UnaryOperator};

use crate::binder::{BoundExpr, ColumnRef};

/// The lake schema a table's catalog columns describe.
///
/// A lake table's schema is written from its catalog columns one for one,
/// so this reproduces what a manifest holds without opening the log. It is
/// what lets a plan reason about the answer a scan will reach before
/// anything is read
pub fn lake_schema_of(te: &zyron_catalog::TableEntry) -> Option<zyron_lake::LakeSchema> {
    let columns: Vec<zyron_lake::LakeColumn> = te
        .columns
        .iter()
        .map(|c| zyron_lake::LakeColumn {
            id: c.id.0 as u32,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
            tz_offset_secs: c.tz_offset_secs,
            max_length: c.max_length.map(|n| n as u32),
            default_expr: c.default_expr.clone(),
        })
        .collect();
    zyron_lake::LakeSchema::new(1, columns).ok()
}

/// Whether a lake scan of this table answers the predicate on stored bytes
/// alone, leaving no row for a filter over decoded values to remove.
///
/// A predicate that lowers exactly is applied to the keep mask before a
/// projected column is decoded, so the columns it reads are the scan's own
/// business. A caller that only needs those columns to evaluate the
/// predicate does not need them at all, and a `COUNT(*)` over a filtered
/// lake table then decodes nothing.
///
/// The scan re-derives this against the schema the manifest holds and
/// falls back to reading those columns itself when the two disagree, so a
/// wrong answer here costs the saving and never the result
pub fn scan_answers_predicate(expr: &BoundExpr, te: &zyron_catalog::TableEntry) -> bool {
    // A foreign table has no lake files here at all, and its access path is
    // chosen ahead of the lake gate
    if !te.lake.is_lake() || te.foreign.is_foreign() {
        return false;
    }
    let Some(lowered) = lower_predicate(expr, &te.columns, &te.cluster.derived) else {
        return false;
    };
    let Some(schema) = lake_schema_of(te) else {
        return false;
    };
    zyron_lake::StoredFilter::lower(&lowered, &schema).is_some_and(|f| f.is_exact())
}

/// Lowers what it can of a scan predicate, one top level conjunct at a
/// time.
///
/// A scan's lowering only has to be implied by the predicate, never
/// equivalent to it: it drops files that provably hold no matching row and
/// prunes rows on stored bytes, and the row filter still decides what is
/// returned. Refusing the whole predicate because one conjunct has no lake
/// form throws away the pruning every other conjunct would have bought,
/// and an equality on a clustered column beside a LIKE is exactly that
/// shape.
///
/// A DML lowering is a different question and stays all or nothing:
/// [`lower_predicate`] records what a delete removes rather than what a
/// scan may skip, and a weaker predicate there would remove the wrong rows.
///
/// Only top level conjuncts split. An arm of a disjunction that does not
/// lower makes the whole disjunction admit everything, which
/// [`lower_predicate`] already reports by refusing it
pub fn lower_scan_predicate(
    expr: &BoundExpr,
    columns: &[ColumnEntry],
    derived: &[DerivedColumnEntry],
) -> Option<LakePredicate> {
    let mut kept: Vec<LakePredicate> = Vec::new();
    let mut conjuncts = Vec::new();
    split_conjuncts_into(expr, &mut conjuncts);
    for conjunct in conjuncts {
        if let Some(lowered) = lower_predicate(conjunct, columns, derived) {
            kept.push(lowered);
        }
    }
    match kept.len() {
        0 => None,
        1 => kept.pop(),
        _ => Some(LakePredicate::And(kept)),
    }
}

/// The top level conjuncts a lake scan answers on stored bytes, and the
/// ones left over for a row filter.
///
/// A conjunct the scan answers exactly is applied to the keep mask before
/// any projected column is decoded, so evaluating it again over decoded
/// values can only agree with what is already decided, and the columns it
/// reads are the scan's own business. What is left over decides both what
/// the row filter still has to run and which columns the projection has to
/// carry.
///
/// One inexact conjunct used to make the whole predicate unanswered, so an
/// equality on a clustered column beside a float range paid for both
/// columns and re-ran both terms.
///
/// The projection rule and the scan build call this, so the columns one
/// withholds are the columns the other leaves unevaluated. A disagreement
/// is still safe, because the scan decodes the predicate's own columns when
/// its projection does not carry them, but it costs the saving
pub fn split_scan_answered(
    expr: &BoundExpr,
    te: &zyron_catalog::TableEntry,
) -> (Vec<BoundExpr>, Vec<BoundExpr>) {
    let mut conjuncts = Vec::new();
    split_conjuncts_into(expr, &mut conjuncts);
    let mut answered = Vec::new();
    let mut residual = Vec::new();
    for conjunct in conjuncts {
        if scan_answers_predicate(conjunct, te) {
            answered.push(conjunct.clone());
        } else {
            residual.push(conjunct.clone());
        }
    }
    (answered, residual)
}

fn split_conjuncts_into<'a>(expr: &'a BoundExpr, out: &mut Vec<&'a BoundExpr>) {
    match expr {
        BoundExpr::Nested(inner) => split_conjuncts_into(inner, out),
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            split_conjuncts_into(left, out);
            split_conjuncts_into(right, out);
        }
        other => out.push(other),
    }
}

/// Lowers a bound predicate over one table. Returns None when the
/// expression has no exact lake equivalent
pub fn lower_predicate(
    expr: &BoundExpr,
    columns: &[ColumnEntry],
    derived: &[DerivedColumnEntry],
) -> Option<LakePredicate> {
    match expr {
        BoundExpr::Nested(inner) => lower_predicate(inner, columns, derived),
        BoundExpr::BinaryOp {
            left, op, right, ..
        } => match op {
            BinaryOperator::And => Some(LakePredicate::And(vec![
                lower_predicate(left, columns, derived)?,
                lower_predicate(right, columns, derived)?,
            ])),
            BinaryOperator::Or => Some(LakePredicate::Or(vec![
                lower_predicate(left, columns, derived)?,
                lower_predicate(right, columns, derived)?,
            ])),
            BinaryOperator::Eq
            | BinaryOperator::Neq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq => lower_comparison(left, *op, right, columns, derived),
            _ => None,
        },
        BoundExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
            ..
        } => Some(lower_predicate(expr, columns, derived)?.negated()),
        BoundExpr::IsNull { expr, negated } => {
            let (col, _) = resolve_column(expr, columns, derived)?;
            Some(if *negated {
                LakePredicate::IsNotNull { column_id: col.id }
            } else {
                LakePredicate::IsNull { column_id: col.id }
            })
        }
        BoundExpr::InList {
            expr,
            list,
            negated,
        } => {
            let (col, fractional_digits) = resolve_column(expr, columns, derived)?;
            let mut values = Vec::with_capacity(list.len());
            for item in list {
                values.push(literal_to_value(item, col, fractional_digits)?);
            }
            let base = LakePredicate::In {
                column_id: col.id,
                values,
            };
            Some(if *negated { base.negated() } else { base })
        }
        BoundExpr::Between {
            expr,
            low,
            high,
            negated,
        } => {
            let (col, fractional_digits) = resolve_column(expr, columns, derived)?;
            let low = literal_to_value(low, col, fractional_digits)?;
            let high = literal_to_value(high, col, fractional_digits)?;
            let base = LakePredicate::And(vec![
                LakePredicate::Compare {
                    column_id: col.id,
                    op: CompareOp::GtEq,
                    value: low,
                },
                LakePredicate::Compare {
                    column_id: col.id,
                    op: CompareOp::LtEq,
                    value: high,
                },
            ]);
            Some(if *negated { base.negated() } else { base })
        }
        _ => None,
    }
}

/// Lowers `<column> <op> <literal>` in either operand order
fn lower_comparison(
    left: &BoundExpr,
    op: BinaryOperator,
    right: &BoundExpr,
    columns: &[ColumnEntry],
    derived: &[DerivedColumnEntry],
) -> Option<LakePredicate> {
    // The column may sit on either side, the operator flips with it
    let (col, fractional_digits, literal, op) = match resolve_column(left, columns, derived) {
        Some((col, p)) => (col, p, right, op),
        None => {
            let (col, p) = resolve_column(right, columns, derived)?;
            (col, p, left, flip(op))
        }
    };
    let value = literal_to_value(literal, col, fractional_digits)?;
    Some(LakePredicate::Compare {
        column_id: col.id,
        op: compare_op(op)?,
        value,
    })
}

fn flip(op: BinaryOperator) -> BinaryOperator {
    match op {
        BinaryOperator::Lt => BinaryOperator::Gt,
        BinaryOperator::Gt => BinaryOperator::Lt,
        BinaryOperator::LtEq => BinaryOperator::GtEq,
        BinaryOperator::GtEq => BinaryOperator::LtEq,
        other => other,
    }
}

fn compare_op(op: BinaryOperator) -> Option<CompareOp> {
    Some(match op {
        BinaryOperator::Eq => CompareOp::Eq,
        BinaryOperator::Neq => CompareOp::NotEq,
        BinaryOperator::Lt => CompareOp::Lt,
        BinaryOperator::LtEq => CompareOp::LtEq,
        BinaryOperator::Gt => CompareOp::Gt,
        BinaryOperator::GtEq => CompareOp::GtEq,
        _ => return None,
    })
}

/// Renders a lowered predicate back to SQL text for the manifest's
/// recorded delete and for diagnostics. The IR is small enough that the
/// rendering is faithful, so what a user reads back is what the delete
/// actually applies
pub fn render_sql(predicate: &LakePredicate, columns: &[ColumnEntry]) -> String {
    let name = |id: u32| -> String {
        columns
            .iter()
            .find(|c| c.id.0 as u32 == id)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| format!("column_{}", id))
    };
    match predicate {
        LakePredicate::Compare {
            column_id,
            op,
            value,
        } => {
            let op = match op {
                CompareOp::Eq => "=",
                CompareOp::NotEq => "<>",
                CompareOp::Lt => "<",
                CompareOp::LtEq => "<=",
                CompareOp::Gt => ">",
                CompareOp::GtEq => ">=",
            };
            format!("{} {} {}", name(*column_id), op, render_value(value))
        }
        LakePredicate::IsNull { column_id } => format!("{} IS NULL", name(*column_id)),
        LakePredicate::IsNotNull { column_id } => format!("{} IS NOT NULL", name(*column_id)),
        LakePredicate::In { column_id, values } => {
            let items: Vec<String> = values.iter().map(render_value).collect();
            format!("{} IN ({})", name(*column_id), items.join(", "))
        }
        LakePredicate::And(children) => join_children(children, " AND ", columns),
        LakePredicate::Or(children) => join_children(children, " OR ", columns),
        LakePredicate::Not(inner) => format!("NOT ({})", render_sql(inner, columns)),
    }
}

fn join_children(children: &[LakePredicate], sep: &str, columns: &[ColumnEntry]) -> String {
    if children.is_empty() {
        // An empty conjunction is true, an empty disjunction is false, but
        // the lowering never produces either, so this is only a guard
        return "TRUE".to_string();
    }
    let parts: Vec<String> = children
        .iter()
        .map(|c| format!("({})", render_sql(c, columns)))
        .collect();
    parts.join(sep)
}

fn render_value(value: &LakeValue) -> String {
    match value {
        LakeValue::Null => "NULL".to_string(),
        LakeValue::Bool(b) => b.to_string(),
        LakeValue::Int(v) => v.to_string(),
        LakeValue::Int128(v) => v.to_string(),
        LakeValue::UInt(v) => v.to_string(),
        LakeValue::UInt128(v) => v.to_string(),
        LakeValue::Float(v) => v.to_string(),
        LakeValue::Str(s) => format!("'{}'", s.replace('\'', "''")),
        LakeValue::Bytes(b) => {
            let mut out = String::with_capacity(3 + b.len() * 2);
            out.push_str("x'");
            for byte in b {
                out.push_str(&format!("{:02x}", byte));
            }
            out.push('\'');
            out
        }
    }
}

/// Column view used by the lowering: the lake addresses columns by the
/// catalog column id, which is what the lake schema carries too
#[derive(Clone, Copy)]
struct LakeColumnRef {
    id: u32,
    type_id: TypeId,
}

/// Resolves an operand to a column of the target table. A reference to a
/// column the table does not have, or any wrapping expression, is not a
/// column for lowering purposes
fn resolve_column(
    expr: &BoundExpr,
    columns: &[ColumnEntry],
    derived: &[DerivedColumnEntry],
) -> Option<(LakeColumnRef, Option<u8>)> {
    match expr {
        BoundExpr::Nested(inner) => resolve_column(inner, columns, derived),
        BoundExpr::ColumnRef(ColumnRef { column_id, .. }) => {
            let entry = columns.iter().find(|c| c.id == *column_id)?;
            Some((
                LakeColumnRef {
                    id: entry.id.0 as u32,
                    type_id: entry.type_id,
                },
                entry.fractional_digits,
            ))
        }
        // An expression the table is clustered by has a column holding its
        // values, so it resolves to that column and everything downstream
        // prunes it as an ordinary column reference
        other => {
            if derived.is_empty() {
                return None;
            }
            let canonical = crate::cluster_expr::canonicalize(other, columns)?;
            let entry = derived
                .iter()
                .find(|d| d.canonical_hash == canonical.canonical_hash)?;
            Some((
                LakeColumnRef {
                    id: entry.column_id,
                    type_id: TypeId::from_u8(entry.type_id)?,
                },
                match entry.fractional_digits {
                    NO_FRACTIONAL_DIGITS => None,
                    digits => Some(digits),
                },
            ))
        }
    }
}

/// The `fractional_digits` byte a mirrored expression column writes when
/// its result declares no precision
const NO_FRACTIONAL_DIGITS: u8 = 0xFF;

/// Converts a literal into the column's storage value domain. Only pairs
/// whose comparison is provably identical to the executor's own lower,
/// everything else refuses so the caller falls back
fn literal_to_value(
    expr: &BoundExpr,
    col: LakeColumnRef,
    fractional_digits: Option<u8>,
) -> Option<LakeValue> {
    let value = match expr {
        BoundExpr::Nested(inner) => return literal_to_value(inner, col, fractional_digits),
        // A typed literal like TIMESTAMP '...' parses as a cast of a
        // string literal. When the cast lands on the column's own type,
        // the conversion below is the same one the executor's cast runs,
        // so the constant lowers instead of defeating every prune that
        // spells its boundary the standard way
        BoundExpr::Cast {
            expr: inner,
            target_type,
            ..
        } if *target_type == col.type_id => {
            return literal_to_value(inner, col, fractional_digits);
        }
        BoundExpr::Literal { value, .. } => value,
        _ => return None,
    };
    // A comparison against NULL is unknown for every row whatever the
    // column type, which the lake predicate models directly
    if matches!(value, LiteralValue::Null) {
        return Some(LakeValue::Null);
    }
    match col.type_id {
        TypeId::Boolean => match value {
            LiteralValue::Boolean(b) => Some(LakeValue::Bool(*b)),
            _ => None,
        },
        // Signed and unsigned integers compare exactly across signedness
        // in the lake value domain, so one integer constant serves both
        TypeId::Int8
        | TypeId::Int16
        | TypeId::Int32
        | TypeId::Int64
        | TypeId::Int128
        | TypeId::UInt8
        | TypeId::UInt16
        | TypeId::UInt32
        | TypeId::UInt64
        | TypeId::UInt128 => match value {
            LiteralValue::Integer(v) => Some(LakeValue::Int(*v)),
            _ => None,
        },
        TypeId::Float32 | TypeId::Float64 => match value {
            LiteralValue::Float(f) => Some(LakeValue::Float(*f)),
            // An integer constant lowers only when f64 holds it exactly
            LiteralValue::Integer(v) if (*v as f64) as i64 == *v => {
                Some(LakeValue::Float(*v as f64))
            }
            _ => None,
        },
        // Timestamps are i64 microseconds. A p>6 column stores i128
        // picoseconds, a different domain, so it does not lower here
        TypeId::Timestamp | TypeId::TimestampTz if fractional_digits.unwrap_or(6) <= 6 => {
            match value {
                LiteralValue::Integer(v) => Some(LakeValue::Int(*v)),
                LiteralValue::String(s) => zyron_common::parse_timestamp_micros(s)
                    .ok()
                    .map(LakeValue::Int),
                _ => None,
            }
        }
        // Byte order equals collation order for these string types
        TypeId::Varchar | TypeId::Text => match value {
            LiteralValue::String(s) => Some(LakeValue::Str(s.clone())),
            _ => None,
        },
        // Everything else needs a domain conversion the lowering does not
        // model exactly: decimal scale, date and time text parsing, uuid
        // and network text forms, blank-padded CHAR, and the opaque
        // variable-length types that carry no ordering at all
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_catalog::{ColumnId, TableId};
    use zyron_lake::{ColumnBounds, PruneDecision, StatsSource};

    fn column(id: u16, name: &str, type_id: TypeId) -> ColumnEntry {
        ColumnEntry {
            id: ColumnId(id),
            table_id: TableId(1),
            name: name.to_string(),
            type_id,
            ordinal: id,
            nullable: true,
            default_expr: None,
            max_length: None,
            fractional_digits: None,
            tz_offset_secs: None,
            element_type: None,
            attrs: Default::default(),
            absent_value: None,
            dropped: false,
        }
    }

    fn columns() -> Vec<ColumnEntry> {
        vec![
            column(0, "id", TypeId::Int64),
            column(1, "name", TypeId::Text),
            column(2, "ts", TypeId::Timestamp),
            column(3, "amount", TypeId::Decimal),
        ]
    }

    fn col_ref(id: u16, type_id: TypeId) -> BoundExpr {
        BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(id),
            type_id,
            nullable: true,
            fractional_digits: None,
        })
    }

    fn int_lit(v: i64) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::Integer(v),
            type_id: TypeId::Int64,
        }
    }

    fn str_lit(s: &str) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::String(s.to_string()),
            type_id: TypeId::Text,
        }
    }

    fn cmp(left: BoundExpr, op: BinaryOperator, right: BoundExpr) -> BoundExpr {
        BoundExpr::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
            type_id: TypeId::Boolean,
        }
    }

    /// A lake table over `cols`, which is all `split_scan_answered` reads
    fn lake_table(cols: Vec<ColumnEntry>) -> zyron_catalog::TableEntry {
        zyron_catalog::TableEntry {
            id: TableId(1),
            schema_id: zyron_catalog::SchemaId(1),
            name: "t".to_string(),
            heap_file_id: 200,
            fsm_file_id: 201,
            columns: cols,
            constraints: vec![],
            created_at: 0,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
            lifecycle: Default::default(),
            columnar: Default::default(),
            dropped_at: None,
            expectations: Vec::new(),
            time_travel_retention_secs: 0,
            lake: zyron_catalog::schema::LakeConfig::lake(),
            cluster: Default::default(),
            foreign: Default::default(),
            schema_epoch: 1,
            schema_epochs: Vec::new(),
            pre_stamp_columns: Vec::new(),
            cdf: Default::default(),
        }
    }

    /// One conjunct the scan cannot answer used to make the whole predicate
    /// its own work, so an equality the scan settles on stored bytes still
    /// paid for its column and ran again over the decoded values.
    ///
    /// The float range is the case that survives every widening: a
    /// `Predicate::Range` is defined over unsigned byte order and ALP
    /// answers one in float order, so nothing is pushed and the term only
    /// prunes zones
    #[test]
    fn a_scan_keeps_the_conjuncts_it_answers_and_hands_back_the_rest() {
        let te = lake_table(vec![
            column(0, "id", TypeId::Int64),
            column(1, "price", TypeId::Float64),
            column(2, "name", TypeId::Text),
        ]);
        let exact = cmp(col_ref(0, TypeId::Int64), BinaryOperator::Eq, int_lit(7));
        let float_range = cmp(
            col_ref(1, TypeId::Float64),
            BinaryOperator::Gt,
            BoundExpr::Literal {
                value: LiteralValue::Float(9.99),
                type_id: TypeId::Float64,
            },
        );

        let (answered, residual) = split_scan_answered(
            &cmp(exact.clone(), BinaryOperator::And, float_range.clone()),
            &te,
        );
        assert_eq!(
            answered.len(),
            1,
            "the equality is answered on stored bytes"
        );
        assert_eq!(
            residual.len(),
            1,
            "the float range is left to the row filter"
        );

        // Both answered leaves nothing for a row filter to do
        let second = cmp(col_ref(0, TypeId::Int64), BinaryOperator::Lt, int_lit(90));
        let (answered, residual) =
            split_scan_answered(&cmp(exact.clone(), BinaryOperator::And, second), &te);
        assert_eq!(answered.len(), 2);
        assert!(residual.is_empty());

        // A term with no lake form at all is residual, and does not stop
        // the equality beside it from being answered. A comparison between
        // two columns is that shape: the lowering is defined over a column
        // against a constant and there is no constant here
        let unlowerable = cmp(
            col_ref(0, TypeId::Int64),
            BinaryOperator::Gt,
            col_ref(1, TypeId::Float64),
        );
        let (answered, residual) =
            split_scan_answered(&cmp(exact.clone(), BinaryOperator::And, unlowerable), &te);
        assert_eq!(answered.len(), 1, "the equality is still answered");
        assert_eq!(residual.len(), 1, "a term with no lake form is residual");

        // A heap table answers none of it
        let heap = lake_table_as_heap(&te);
        let (answered, residual) =
            split_scan_answered(&cmp(exact, BinaryOperator::And, float_range), &heap);
        assert!(answered.is_empty());
        assert_eq!(residual.len(), 2);
    }

    fn lake_table_as_heap(te: &zyron_catalog::TableEntry) -> zyron_catalog::TableEntry {
        let mut heap = te.clone();
        heap.lake = Default::default();
        heap
    }

    struct OneColumn {
        column_id: u32,
        bounds: ColumnBounds,
    }

    impl StatsSource for OneColumn {
        fn bounds(&self, column_id: u32) -> Option<&ColumnBounds> {
            (column_id == self.column_id).then_some(&self.bounds)
        }
    }

    #[test]
    fn test_lowers_comparison_in_either_operand_order() {
        let cols = columns();
        let direct = lower_predicate(
            &cmp(col_ref(0, TypeId::Int64), BinaryOperator::Lt, int_lit(50)),
            &cols,
            &[],
        )
        .expect("lowers");
        assert_eq!(
            direct,
            LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(50),
            }
        );
        // The operator flips with the operands
        let flipped = lower_predicate(
            &cmp(int_lit(50), BinaryOperator::Lt, col_ref(0, TypeId::Int64)),
            &cols,
            &[],
        )
        .expect("lowers");
        assert_eq!(
            flipped,
            LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Gt,
                value: LakeValue::Int(50),
            }
        );
    }

    #[test]
    fn test_lowers_and_or_not_null_in_between() {
        let cols = columns();
        let and_or = lower_predicate(
            &cmp(
                cmp(col_ref(0, TypeId::Int64), BinaryOperator::GtEq, int_lit(1)),
                BinaryOperator::And,
                cmp(
                    col_ref(1, TypeId::Text),
                    BinaryOperator::Eq,
                    str_lit("alice"),
                ),
            ),
            &cols,
            &[],
        )
        .expect("lowers");
        assert!(matches!(and_or, LakePredicate::And(ref v) if v.len() == 2));

        let is_null = lower_predicate(
            &BoundExpr::IsNull {
                expr: Box::new(col_ref(1, TypeId::Text)),
                negated: true,
            },
            &cols,
            &[],
        )
        .expect("lowers");
        assert_eq!(is_null, LakePredicate::IsNotNull { column_id: 1 });

        let in_list = lower_predicate(
            &BoundExpr::InList {
                expr: Box::new(col_ref(0, TypeId::Int64)),
                list: vec![int_lit(1), int_lit(2)],
                negated: false,
            },
            &cols,
            &[],
        )
        .expect("lowers");
        assert_eq!(
            in_list,
            LakePredicate::In {
                column_id: 0,
                values: vec![LakeValue::Int(1), LakeValue::Int(2)],
            }
        );

        let between = lower_predicate(
            &BoundExpr::Between {
                expr: Box::new(col_ref(0, TypeId::Int64)),
                low: Box::new(int_lit(10)),
                high: Box::new(int_lit(20)),
                negated: true,
            },
            &cols,
            &[],
        )
        .expect("lowers");
        // NOT BETWEEN is exactly the disjunction of the strict outsides
        assert_eq!(
            between,
            LakePredicate::Or(vec![
                LakePredicate::Compare {
                    column_id: 0,
                    op: CompareOp::Lt,
                    value: LakeValue::Int(10),
                },
                LakePredicate::Compare {
                    column_id: 0,
                    op: CompareOp::Gt,
                    value: LakeValue::Int(20),
                },
            ])
        );
    }

    #[test]
    fn test_timestamp_string_literal_uses_the_executor_parse() {
        let cols = columns();
        let lowered = lower_predicate(
            &cmp(
                col_ref(2, TypeId::Timestamp),
                BinaryOperator::Lt,
                str_lit("2026-01-01 00:00:00"),
            ),
            &cols,
            &[],
        )
        .expect("lowers");
        let expected = zyron_common::parse_timestamp_micros("2026-01-01 00:00:00").expect("parses");
        assert_eq!(
            lowered,
            LakePredicate::Compare {
                column_id: 2,
                op: CompareOp::Lt,
                value: LakeValue::Int(expected),
            }
        );
        // A retention delete over a file entirely below the cut covers it
        let stats = OneColumn {
            column_id: 2,
            bounds: ColumnBounds {
                min: Some(LakeValue::Int(expected - 10_000)),
                max: Some(LakeValue::Int(expected - 1)),
                null_count: 0,
                row_count: 100,
            },
        };
        assert_eq!(lowered.prune(&stats), PruneDecision::FullyCovers);
    }

    #[test]
    fn test_refuses_what_it_cannot_prove() {
        let cols = columns();
        // Decimal carries a scale the lowering does not model
        assert!(
            lower_predicate(
                &cmp(col_ref(3, TypeId::Decimal), BinaryOperator::Eq, int_lit(5)),
                &cols,
                &[],
            )
            .is_none()
        );
        // A column of another table is not a column of this one
        assert!(
            lower_predicate(
                &cmp(col_ref(99, TypeId::Int64), BinaryOperator::Eq, int_lit(5)),
                &cols,
                &[],
            )
            .is_none()
        );
        // Column to column is not a constant comparison
        assert!(
            lower_predicate(
                &cmp(
                    col_ref(0, TypeId::Int64),
                    BinaryOperator::Eq,
                    col_ref(0, TypeId::Int64)
                ),
                &cols,
                &[],
            )
            .is_none()
        );
        // One unlowerable conjunct refuses the whole predicate, a partial
        // lowering would select the wrong rows for a delete
        assert!(
            lower_predicate(
                &cmp(
                    cmp(col_ref(0, TypeId::Int64), BinaryOperator::Lt, int_lit(5)),
                    BinaryOperator::And,
                    cmp(col_ref(3, TypeId::Decimal), BinaryOperator::Eq, int_lit(5)),
                ),
                &cols,
                &[],
            )
            .is_none()
        );
        // LIKE has no lake equivalent
        assert!(
            lower_predicate(
                &BoundExpr::Like {
                    expr: Box::new(col_ref(1, TypeId::Text)),
                    pattern: Box::new(str_lit("a%")),
                    negated: false,
                },
                &cols,
                &[],
            )
            .is_none()
        );
    }

    #[test]
    fn test_render_sql_round_trips_the_shape() {
        let cols = columns();
        let lowered = lower_predicate(
            &cmp(
                cmp(col_ref(0, TypeId::Int64), BinaryOperator::Lt, int_lit(50)),
                BinaryOperator::And,
                BoundExpr::IsNull {
                    expr: Box::new(col_ref(1, TypeId::Text)),
                    negated: false,
                },
            ),
            &cols,
            &[],
        )
        .expect("lowers");
        assert_eq!(render_sql(&lowered, &cols), "(id < 50) AND (name IS NULL)");

        let quoted = lower_predicate(
            &cmp(
                col_ref(1, TypeId::Text),
                BinaryOperator::Eq,
                str_lit("it's"),
            ),
            &cols,
            &[],
        )
        .expect("lowers");
        assert_eq!(render_sql(&quoted, &cols), "name = 'it''s'");
    }

    #[test]
    fn test_null_literal_comparison_selects_nothing() {
        let cols = columns();
        let lowered = lower_predicate(
            &cmp(
                col_ref(0, TypeId::Int64),
                BinaryOperator::Eq,
                BoundExpr::Literal {
                    value: LiteralValue::Null,
                    type_id: TypeId::Null,
                },
            ),
            &cols,
            &[],
        )
        .expect("lowers");
        let stats = OneColumn {
            column_id: 0,
            bounds: ColumnBounds {
                min: Some(LakeValue::Int(1)),
                max: Some(LakeValue::Int(100)),
                null_count: 0,
                row_count: 10,
            },
        };
        assert_eq!(lowered.prune(&stats), PruneDecision::CannotMatch);
    }

    /// A predicate over a clustered expression lowers to a reference to the
    /// column holding that expression's values. Everything downstream then
    /// prunes it as an ordinary column, which is the whole reason an
    /// expression cluster key is given a column
    #[test]
    fn test_predicate_over_a_clustered_expression_lowers_to_its_column() {
        use zyron_catalog::schema::DerivedColumnEntry;

        let cols = columns();
        let ts_day = crate::cluster_expr::canonicalize(
            &BoundExpr::Function {
                name: "date_trunc".into(),
                args: vec![
                    BoundExpr::Literal {
                        value: LiteralValue::String("day".into()),
                        type_id: TypeId::Varchar,
                    },
                    col_ref(2, TypeId::Timestamp),
                ],
                return_type: TypeId::Timestamp,
                distinct: false,
            },
            &cols,
        )
        .expect("clusterable");
        let registry = vec![DerivedColumnEntry {
            column_id: 40,
            canonical_hash: ts_day.canonical_hash,
            type_id: TypeId::Timestamp as u8,
            fractional_digits: 0xFF,
            sql: "expr".into(),
        }];

        // The same expression written with a different keyword case still
        // finds the column, because identity is the canonical hash
        let predicate = lower_predicate(
            &cmp(
                BoundExpr::Function {
                    name: "DATE_TRUNC".into(),
                    args: vec![
                        BoundExpr::Literal {
                            value: LiteralValue::String("DAY".into()),
                            type_id: TypeId::Varchar,
                        },
                        col_ref(2, TypeId::Timestamp),
                    ],
                    return_type: TypeId::Timestamp,
                    distinct: false,
                },
                BinaryOperator::Eq,
                int_lit(1_700_000_000_000_000),
            ),
            &cols,
            &registry,
        )
        .expect("lowers through the derived column");
        assert_eq!(
            predicate,
            LakePredicate::Compare {
                column_id: 40,
                op: CompareOp::Eq,
                value: LakeValue::Int(1_700_000_000_000_000),
            }
        );

        // With no registry the same predicate has no lake equivalent, which
        // is what keeps a table that is not clustered by the expression from
        // pruning on statistics it does not have
        assert!(
            lower_predicate(
                &cmp(
                    BoundExpr::Function {
                        name: "date_trunc".into(),
                        args: vec![
                            BoundExpr::Literal {
                                value: LiteralValue::String("day".into()),
                                type_id: TypeId::Varchar,
                            },
                            col_ref(2, TypeId::Timestamp),
                        ],
                        return_type: TypeId::Timestamp,
                        distinct: false,
                    },
                    BinaryOperator::Eq,
                    int_lit(1_700_000_000_000_000),
                ),
                &cols,
                &[],
            )
            .is_none()
        );

        // A different expression does not match the registered one
        assert!(
            lower_predicate(
                &cmp(
                    BoundExpr::Function {
                        name: "date_trunc".into(),
                        args: vec![
                            BoundExpr::Literal {
                                value: LiteralValue::String("month".into()),
                                type_id: TypeId::Varchar,
                            },
                            col_ref(2, TypeId::Timestamp),
                        ],
                        return_type: TypeId::Timestamp,
                        distinct: false,
                    },
                    BinaryOperator::Eq,
                    int_lit(1_700_000_000_000_000),
                ),
                &cols,
                &registry,
            )
            .is_none()
        );
    }

    /// A derived column over a high precision timestamp must carry that
    /// precision, or its constant lowers into the wrong domain.
    ///
    /// A TIMESTAMP(p) with p greater than six stores i128 picoseconds, and
    /// the lowering deliberately refuses that domain because it models
    /// microseconds. Mirroring the precision is what lets it refuse. With
    /// the precision lost the same predicate lowers as microseconds and is
    /// then compared against picosecond bounds, which prunes files holding
    /// rows that match
    #[test]
    fn test_derived_column_precision_decides_whether_a_constant_may_lower() {
        use zyron_catalog::schema::DerivedColumnEntry;

        let cols = columns();
        let day_of_ts = |name: &str, unit: &str| BoundExpr::Function {
            name: name.to_string(),
            args: vec![
                BoundExpr::Literal {
                    value: LiteralValue::String(unit.to_string()),
                    type_id: TypeId::Varchar,
                },
                col_ref(2, TypeId::Timestamp),
            ],
            return_type: TypeId::Timestamp,
            distinct: false,
        };
        let canonical = crate::cluster_expr::canonicalize(&day_of_ts("date_trunc", "day"), &cols)
            .expect("clusterable");
        let registry = |fractional_digits: u8| {
            vec![DerivedColumnEntry {
                column_id: 40,
                canonical_hash: canonical.canonical_hash,
                type_id: TypeId::Timestamp as u8,
                fractional_digits,
                sql: "date_trunc('day', ts)".into(),
            }]
        };
        let predicate = || {
            cmp(
                day_of_ts("date_trunc", "day"),
                BinaryOperator::Eq,
                str_lit("2026-08-18 00:00:00"),
            )
        };

        // Picosecond domain, so the microsecond lowering must decline and
        // the scan keeps every file rather than pruning on a mismatched scale
        assert!(
            lower_predicate(&predicate(), &cols, &registry(9)).is_none(),
            "a picosecond derived column must not lower a microsecond constant"
        );
        assert!(lower_predicate(&predicate(), &cols, &registry(12)).is_none());

        // Microsecond domain, where the constant and the stored values agree
        let lowered = lower_predicate(&predicate(), &cols, &registry(6))
            .expect("a microsecond derived column lowers");
        assert!(matches!(
            lowered,
            LakePredicate::Compare {
                column_id: 40,
                op: CompareOp::Eq,
                ..
            }
        ));
    }

    /// A decimal derived column has a scale the lowering does not model, so
    /// it lowers nothing at all. Pinned so a later decimal arm cannot be
    /// added without deciding what its scale means
    #[test]
    fn test_decimal_derived_column_lowers_no_constant() {
        use zyron_catalog::schema::DerivedColumnEntry;

        let cols = columns();
        let rounded = BoundExpr::Function {
            name: "round".to_string(),
            args: vec![col_ref(3, TypeId::Decimal)],
            return_type: TypeId::Decimal,
            distinct: false,
        };
        let canonical = crate::cluster_expr::canonicalize(&rounded, &cols).expect("clusterable");
        for scale in [0u8, 2, 38] {
            let registry = vec![DerivedColumnEntry {
                column_id: 41,
                canonical_hash: canonical.canonical_hash,
                type_id: TypeId::Decimal as u8,
                fractional_digits: scale,
                sql: "expr".into(),
            }];
            assert!(
                lower_predicate(
                    &cmp(rounded.clone(), BinaryOperator::Eq, int_lit(5)),
                    &cols,
                    &registry,
                )
                .is_none(),
                "a decimal constant has a scale the lowering does not model"
            );
        }
    }
}
