//! Rejecting rows from zone maps and encoded bytes, before any decode.
//!
//! File pruning answers from the manifest and never opens a file. What is
//! left is a file the predicate might match, and the next two questions
//! are how much of it to read and how much of that to decode.
//!
//! A zone covers 1024 rows and records their bounds, and a file's own
//! bounds are the union of its zones, so a file can admit a range that no
//! single zone admits. When that happens the whole file is rejected for
//! the cost of its zone region, with no payload read at all.
//!
//! Past that, the predicate is answered on the encoded bytes. Dictionary,
//! run length and constant segments resolve a term from their compact
//! form, so a column whose decoded size is orders of magnitude larger
//! than its encoded size never materializes the difference.
//!
//! Everything here produces a superset of the matching rows and never a
//! subset. The exact row filter still runs on what survives, so a term
//! that cannot be lowered costs pruning and never correctness. Two things
//! are deliberately not lowered:
//!
//! * a float range into the encoding. Stat slots order floats correctly
//!   now, so a float range still rejects zones, but `Predicate::Range` is
//!   defined over unsigned byte order and ALP answers one in float order,
//!   so the encodings do not agree on it. Float equality is byte equality
//!   and every encoding agrees on that, so it is pushed
//! Nothing else is held back. A variable-length column prunes zones from
//! the prefix its slots hold, and a term against a value longer than that
//! prefix is decided by the exact filter on whatever survives.
//!
//! `<>` and `NOT IN` are lowered as the equality mask inverted. A null
//! row's slot is zero-filled and lands on the keep side of that
//! inversion, which is the safe direction: the exact filter removes it,
//! and SQL says a null satisfies no comparison anyway.

use zyron_common::curve::{CellFamily, cell_family};
use zyron_common::{TypeId, ZyronError};

use zyron_storage::columnar::{STAT_VALUE_SIZE, ZONE_MAP_BATCH_SIZE, ZoneMapEntry};
use zyron_storage::columnar::{
    SlotOrder, compare_stat_slots_typed, compare_value_to_slot, slot_order,
};
use zyron_storage::encoding::Predicate;

use crate::cells::value_to_cell;
use crate::predicate::{CompareOp, LakePredicate, LakeValue};
use crate::schema::LakeSchema;

/// Widest column a numeric range is lowered for. Sixteen bytes is every
/// integer-backed type the engine has, and the stat slot holds thirty two
const MAX_RANGE_WIDTH: usize = 16;

/// A lake predicate lowered onto the bytes a data file stores.
///
/// Built once per scan, because the lowering depends on the predicate and
/// the schema and not on any file, then applied to every file the manifest
/// did not already reject
#[derive(Debug, Clone, PartialEq)]
pub struct StoredFilter {
    root: StoredNode,
}

#[derive(Debug, Clone, PartialEq)]
enum StoredNode {
    /// Nothing was lowered, so every row stands
    All,
    /// The term is provably empty
    Nothing,
    Leaf(Leaf),
    And(Vec<StoredNode>),
    Or(Vec<StoredNode>),
}

#[derive(Debug, Clone, PartialEq)]
struct Leaf {
    column_id: u32,
    /// Fixed cell width, zero for a variable-length column
    value_size: usize,
    /// How the column's stat slots compare, which is not the same for an
    /// unsigned integer, a two's complement one and a float
    order: SlotOrder,
    /// Zone map slots order this column the way its values order. False
    /// for a variable-length column, whose slots compare from their last
    /// byte down rather than lexicographically
    zone_prunable: bool,
    /// What the term admits, in the column's value order, for the zone
    /// map check
    admits: Admits,
    /// The same term over stored byte order, empty when it cannot be
    /// pushed into the encoding
    pushdown: Vec<OwnedPredicate>,
    /// The term selects what the pushdown does not. Set by `<>` and
    /// `NOT IN`, whose mask is the equality mask inverted
    invert: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum Admits {
    /// Cells equal to one of these
    Values(Vec<Vec<u8>>),
    /// One inclusive interval, None unbounded on that side
    Interval(Option<Vec<u8>>, Option<Vec<u8>>),
}

#[derive(Debug, Clone, PartialEq)]
enum OwnedPredicate {
    AnyOf(Vec<Vec<u8>>),
    Range {
        low: Option<Vec<u8>>,
        high: Option<Vec<u8>>,
    },
}

impl StoredFilter {
    /// Lowers a predicate against the schema that types its columns.
    ///
    /// Returns None when nothing could be lowered, which saves every file
    /// the zone map read a filter that admits everything would cost
    pub fn lower(predicate: &LakePredicate, schema: &LakeSchema) -> Option<Self> {
        let root = lower_node(predicate, schema, false);
        match root {
            StoredNode::All => None,
            root => Some(Self { root }),
        }
    }

    /// Column ids the filter reads, so a caller can see what it touches
    pub fn columns(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        collect_columns(&self.root, &mut ids);
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

fn collect_columns(node: &StoredNode, ids: &mut Vec<u32>) {
    match node {
        StoredNode::Leaf(leaf) => ids.push(leaf.column_id),
        StoredNode::And(children) | StoredNode::Or(children) => {
            for c in children {
                collect_columns(c, ids);
            }
        }
        StoredNode::All | StoredNode::Nothing => {}
    }
}

fn lower_node(predicate: &LakePredicate, schema: &LakeSchema, negated: bool) -> StoredNode {
    match predicate {
        LakePredicate::Not(inner) => lower_node(inner, schema, !negated),
        LakePredicate::Compare {
            column_id,
            op,
            value,
        } => {
            let op = if negated { op.negated() } else { *op };
            lower_compare(*column_id, op, value, schema)
        }
        // A null-shaped term is answered by the null bitmap the decode
        // already returns, so lowering it would read the same bits twice
        LakePredicate::IsNull { .. } | LakePredicate::IsNotNull { .. } => StoredNode::All,
        LakePredicate::In { column_id, values } if !negated => {
            if values.is_empty() {
                return StoredNode::Nothing;
            }
            let Some(shape) = column_shape(*column_id, schema) else {
                return StoredNode::All;
            };
            let mut cells = Vec::with_capacity(values.len());
            for value in values {
                match value_to_cell(shape.physical, shape.value_size, value) {
                    // One member with no provable stored form makes the
                    // whole membership unprovable, since the rows it would
                    // have admitted must not be dropped
                    None => return StoredNode::All,
                    Some(cell) => cells.push(cell.as_slice().to_vec()),
                }
            }
            StoredNode::Leaf(shape.equality_leaf(cells))
        }
        // NOT IN is the membership mask inverted. A member with no
        // provable stored form is skipped, which only widens the result
        LakePredicate::In { column_id, values } => {
            let Some(shape) = column_shape(*column_id, schema) else {
                return StoredNode::All;
            };
            let cells: Vec<Vec<u8>> = values
                .iter()
                .filter_map(|v| {
                    value_to_cell(shape.physical, shape.value_size, v)
                        .map(|c| c.as_slice().to_vec())
                })
                .collect();
            if cells.is_empty() {
                // NOT IN () excludes nothing
                return StoredNode::All;
            }
            StoredNode::Leaf(shape.inverted_leaf(cells))
        }
        LakePredicate::And(children) | LakePredicate::Or(children) => {
            let conjunction = matches!(predicate, LakePredicate::And(_)) != negated;
            let lowered: Vec<StoredNode> = children
                .iter()
                .map(|c| lower_node(c, schema, negated))
                .collect();
            if conjunction {
                // An arm that lowered to nothing is dropped, which keeps
                // the result a superset
                if lowered.contains(&StoredNode::Nothing) {
                    return StoredNode::Nothing;
                }
                let kept: Vec<StoredNode> = lowered
                    .into_iter()
                    .filter(|n| *n != StoredNode::All)
                    .collect();
                match kept.len() {
                    0 => StoredNode::All,
                    1 => kept.into_iter().next().unwrap_or(StoredNode::All),
                    _ => StoredNode::And(kept),
                }
            } else {
                // A disjunction is only as good as its weakest arm: one
                // arm that admits everything makes the whole term admit
                // everything
                if lowered.contains(&StoredNode::All) {
                    return StoredNode::All;
                }
                let kept: Vec<StoredNode> = lowered
                    .into_iter()
                    .filter(|n| *n != StoredNode::Nothing)
                    .collect();
                match kept.len() {
                    0 => StoredNode::Nothing,
                    1 => kept.into_iter().next().unwrap_or(StoredNode::All),
                    _ => StoredNode::Or(kept),
                }
            }
        }
    }
}

/// What one column's stored bytes look like, or None when they carry no
/// order this can use
struct ColumnShape {
    column_id: u32,
    physical: TypeId,
    value_size: usize,
    order: SlotOrder,
    zone_prunable: bool,
    varlen: bool,
    /// Ranges over this column are lowered as bytes rather than numbers
    float: bool,
}

impl ColumnShape {
    fn equality_leaf(&self, cells: Vec<Vec<u8>>) -> Leaf {
        Leaf {
            column_id: self.column_id,
            value_size: self.value_size,
            order: self.order,
            zone_prunable: self.zone_prunable,
            admits: Admits::Values(cells.clone()),
            pushdown: vec![OwnedPredicate::AnyOf(cells)],
            invert: false,
        }
    }

    /// A leaf that keeps everything the values do not.
    ///
    /// A member with no provable stored form is dropped rather than
    /// refusing the whole term: leaving one out makes the equality mask
    /// smaller and its inverse larger, which keeps more rows than
    /// necessary and is the safe direction
    fn inverted_leaf(&self, cells: Vec<Vec<u8>>) -> Leaf {
        Leaf {
            column_id: self.column_id,
            value_size: self.value_size,
            order: self.order,
            zone_prunable: self.zone_prunable,
            admits: Admits::Values(cells.clone()),
            pushdown: vec![OwnedPredicate::AnyOf(cells)],
            invert: true,
        }
    }
}

fn column_shape(column_id: u32, schema: &LakeSchema) -> Option<ColumnShape> {
    let column = schema.column_by_id(column_id)?;
    let physical = column.physical_type_id();
    let value_size = physical.fixed_size().unwrap_or(0);
    let family = cell_family(physical);
    let (varlen, float) = match family {
        CellFamily::SignedInt | CellFamily::UnsignedInt | CellFamily::Bool => (false, false),
        CellFamily::Str | CellFamily::Bytes => (true, false),
        CellFamily::Float => (false, true),
        // An unordered family has no comparison at all
        CellFamily::Unordered => return None,
    };
    if varlen != (value_size == 0) {
        // A fixed-width byte family, such as a UUID, compares by its bytes
        // in both directions, which is neither of the two shapes below
        return None;
    }
    Some(ColumnShape {
        column_id,
        physical,
        value_size,
        order: slot_order(physical),
        // A variable-length slot holds a prefix compared from its first
        // byte, with the maximum rounded up so it stays a bound
        zone_prunable: varlen || (1..=STAT_VALUE_SIZE).contains(&value_size),
        varlen,
        float,
    })
}

fn lower_compare(
    column_id: u32,
    op: CompareOp,
    value: &LakeValue,
    schema: &LakeSchema,
) -> StoredNode {
    let Some(shape) = column_shape(column_id, schema) else {
        return StoredNode::All;
    };
    match op {
        CompareOp::Eq => match equality_cells(&shape, value) {
            Some(cells) => StoredNode::Leaf(shape.equality_leaf(cells)),
            None => StoredNode::All,
        },
        // The mask is the equality mask inverted, which keeps every row
        // the constant does not pin
        CompareOp::NotEq => match equality_cells(&shape, value) {
            Some(cells) => StoredNode::Leaf(shape.inverted_leaf(cells)),
            None => StoredNode::All,
        },
        _ if shape.varlen => lower_varlen_range(&shape, op, value),
        _ if shape.float => lower_float_range(&shape, op, value),
        _ => lower_numeric_range(&shape, op, value),
    }
}

/// Every stored cell a constant can equal.
///
/// One for almost every type. Zero is the exception: a float has two
/// spellings of it and they compare equal, so a term on either has to
/// admit both or it would drop rows holding the other
fn equality_cells(shape: &ColumnShape, value: &LakeValue) -> Option<Vec<Vec<u8>>> {
    let cell = value_to_cell(shape.physical, shape.value_size, value)?;
    let mut cells = vec![cell.as_slice().to_vec()];
    if shape.float
        && let LakeValue::Float(v) = value
        && *v == 0.0
    {
        let other = if v.is_sign_negative() {
            0.0f64
        } else {
            -0.0f64
        };
        if let Some(cell) =
            value_to_cell(shape.physical, shape.value_size, &LakeValue::Float(other))
        {
            let bytes = cell.as_slice().to_vec();
            if !cells.contains(&bytes) {
                cells.push(bytes);
            }
        }
    }
    Some(cells)
}

/// A range over a float column.
///
/// Stat slots order floats by value, so a bound rejects zones directly
/// with no decomposition. It is not pushed into the encoding: a
/// `Predicate::Range` is defined over unsigned byte order and ALP answers
/// one in float order, so the encodings disagree on exactly this shape.
/// The bound stays inclusive for a strict operator, which admits the rows
/// equal to the constant and leaves them to the exact filter.
///
/// A NaN bound decides nothing, since no comparison against it is ever
/// true, so it prunes nothing rather than pruning on a value with no
/// position in the order
fn lower_float_range(shape: &ColumnShape, op: CompareOp, value: &LakeValue) -> StoredNode {
    if matches!(value, LakeValue::Float(v) if v.is_nan()) {
        return StoredNode::All;
    }
    let Some(cell) = value_to_cell(shape.physical, shape.value_size, value) else {
        return StoredNode::All;
    };
    let bound = cell.as_slice().to_vec();
    let (low, high) = match op {
        CompareOp::Lt | CompareOp::LtEq => (None, Some(bound)),
        CompareOp::Gt | CompareOp::GtEq => (Some(bound), None),
        CompareOp::Eq | CompareOp::NotEq => return StoredNode::All,
    };
    if !shape.zone_prunable {
        return StoredNode::All;
    }
    StoredNode::Leaf(Leaf {
        column_id: shape.column_id,
        value_size: shape.value_size,
        order: shape.order,
        zone_prunable: true,
        admits: Admits::Interval(low, high),
        pushdown: Vec::new(),
        invert: false,
    })
}

/// A range over a variable-length column, whose stored bytes are ordered
/// lexicographically by the same rule its values are.
///
/// The bound stays inclusive even for a strict operator, because a byte
/// string has no predecessor to subtract. That admits the rows equal to
/// the constant, which the exact filter then removes
fn lower_varlen_range(shape: &ColumnShape, op: CompareOp, value: &LakeValue) -> StoredNode {
    let Some(cell) = value_to_cell(shape.physical, shape.value_size, value) else {
        return StoredNode::All;
    };
    let bound = cell.as_slice().to_vec();
    let (low, high) = match op {
        CompareOp::Lt | CompareOp::LtEq => (None, Some(bound)),
        CompareOp::Gt | CompareOp::GtEq => (Some(bound), None),
        CompareOp::Eq | CompareOp::NotEq => return StoredNode::All,
    };
    StoredNode::Leaf(Leaf {
        column_id: shape.column_id,
        value_size: 0,
        order: shape.order,
        zone_prunable: shape.zone_prunable,
        admits: Admits::Interval(low.clone(), high.clone()),
        pushdown: vec![OwnedPredicate::Range { low, high }],
        invert: false,
    })
}

/// Which side of a column's domain a constant fell on.
///
/// A bound the column cannot hold still decides the term: everything is
/// below `i64::MAX + 1`, and nothing is above it
#[derive(Clone, Copy)]
enum Placed<T> {
    Below,
    In(T),
    Above,
}

/// A range over an integer-backed column, at any width its stat slots
/// can hold.
///
/// The interval is placed in value space, where a strict operator moves
/// the bound by one, then clamped to what the column can hold. Pushdown
/// bounds are the low `width` bytes of each end, which is the stored
/// cell: for a negative value that is its two's complement reading, which
/// is exactly where the encodings compare it, and it is why a signed
/// interval spanning zero needs two ranges rather than one
fn lower_numeric_range(shape: &ColumnShape, op: CompareOp, value: &LakeValue) -> StoredNode {
    let width = shape.value_size;
    if !(1..=MAX_RANGE_WIDTH).contains(&width) {
        return StoredNode::All;
    }
    if shape.order == SlotOrder::TwosComplement {
        lower_signed_range(shape, width, op, value)
    } else {
        lower_unsigned_range(shape, width, op, value)
    }
}

fn lower_unsigned_range(
    shape: &ColumnShape,
    width: usize,
    op: CompareOp,
    value: &LakeValue,
) -> StoredNode {
    let umax: u128 = if width >= 16 {
        u128::MAX
    } else {
        (1u128 << (8 * width)) - 1
    };
    let placed = match value {
        LakeValue::Bool(b) => Placed::In(*b as u128),
        // A negative constant is below every value an unsigned column holds
        LakeValue::Int(v) if *v < 0 => Placed::Below,
        LakeValue::Int128(v) if *v < 0 => Placed::Below,
        LakeValue::Int(v) => Placed::In(*v as u128),
        LakeValue::Int128(v) => Placed::In(*v as u128),
        LakeValue::UInt(v) => Placed::In(*v as u128),
        LakeValue::UInt128(v) => Placed::In(*v),
        _ => return StoredNode::All,
    };
    let placed = match placed {
        Placed::In(k) if k > umax => Placed::Above,
        other => other,
    };

    let interval = match (op, placed) {
        (CompareOp::Lt | CompareOp::LtEq, Placed::Below) => None,
        (CompareOp::Lt | CompareOp::LtEq, Placed::Above) => Some((0, umax)),
        (CompareOp::Lt, Placed::In(k)) => k.checked_sub(1).map(|hi| (0, hi)),
        (CompareOp::LtEq, Placed::In(k)) => Some((0, k)),
        (CompareOp::Gt | CompareOp::GtEq, Placed::Below) => Some((0, umax)),
        (CompareOp::Gt | CompareOp::GtEq, Placed::Above) => None,
        (CompareOp::Gt, Placed::In(k)) => (k < umax).then(|| (k + 1, umax)),
        (CompareOp::GtEq, Placed::In(k)) => Some((k, umax)),
        (CompareOp::Eq | CompareOp::NotEq, _) => return StoredNode::All,
    };
    let Some((lo, hi)) = interval else {
        return StoredNode::Nothing;
    };
    if lo == 0 && hi == umax {
        return StoredNode::All;
    }

    let low = (lo > 0).then(|| unsigned_le_bytes(lo, width));
    let high = (hi < umax).then(|| unsigned_le_bytes(hi, width));
    StoredNode::Leaf(Leaf {
        column_id: shape.column_id,
        value_size: width,
        order: shape.order,
        zone_prunable: shape.zone_prunable,
        admits: Admits::Interval(low.clone(), high.clone()),
        pushdown: vec![OwnedPredicate::Range { low, high }],
        invert: false,
    })
}

fn lower_signed_range(
    shape: &ColumnShape,
    width: usize,
    op: CompareOp,
    value: &LakeValue,
) -> StoredNode {
    let (smin, smax) = if width >= 16 {
        (i128::MIN, i128::MAX)
    } else {
        let bits = 8 * width as u32;
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    };
    let placed = match value {
        LakeValue::Bool(b) => Placed::In(*b as i128),
        LakeValue::Int(v) => Placed::In(*v as i128),
        LakeValue::Int128(v) => Placed::In(*v),
        LakeValue::UInt(v) => Placed::In(*v as i128),
        // Past what a signed value can express, so above every one of them
        LakeValue::UInt128(v) => match i128::try_from(*v) {
            Ok(x) => Placed::In(x),
            Err(_) => Placed::Above,
        },
        _ => return StoredNode::All,
    };
    let placed = match placed {
        Placed::In(k) if k < smin => Placed::Below,
        Placed::In(k) if k > smax => Placed::Above,
        other => other,
    };

    let interval = match (op, placed) {
        (CompareOp::Lt | CompareOp::LtEq, Placed::Below) => None,
        (CompareOp::Lt | CompareOp::LtEq, Placed::Above) => Some((smin, smax)),
        (CompareOp::Lt, Placed::In(k)) => (k > smin).then(|| (smin, k - 1)),
        (CompareOp::LtEq, Placed::In(k)) => Some((smin, k)),
        (CompareOp::Gt | CompareOp::GtEq, Placed::Below) => Some((smin, smax)),
        (CompareOp::Gt | CompareOp::GtEq, Placed::Above) => None,
        (CompareOp::Gt, Placed::In(k)) => (k < smax).then(|| (k + 1, smax)),
        (CompareOp::GtEq, Placed::In(k)) => Some((k, smax)),
        (CompareOp::Eq | CompareOp::NotEq, _) => return StoredNode::All,
    };
    let Some((lo, hi)) = interval else {
        return StoredNode::Nothing;
    };
    if lo == smin && hi == smax {
        return StoredNode::All;
    }

    let bounded_low = (lo > smin).then(|| le_bytes(lo, width));
    let bounded_high = (hi < smax).then(|| le_bytes(hi, width));

    // Two's complement puts every negative above every non-negative in the
    // unsigned reading the encodings compare in, so an interval that spans
    // zero is two contiguous ranges there
    let mut pushdown = Vec::with_capacity(2);
    if lo < 0 {
        let end = hi.min(-1);
        pushdown.push(OwnedPredicate::Range {
            low: Some(le_bytes(lo, width)),
            high: Some(le_bytes(end, width)),
        });
    }
    if hi >= 0 {
        pushdown.push(OwnedPredicate::Range {
            low: Some(le_bytes(lo.max(0), width)),
            high: Some(le_bytes(hi, width)),
        });
    }

    StoredNode::Leaf(Leaf {
        column_id: shape.column_id,
        value_size: width,
        order: shape.order,
        zone_prunable: shape.zone_prunable,
        admits: Admits::Interval(bounded_low, bounded_high),
        pushdown,
        invert: false,
    })
}

/// The low `width` bytes of an unsigned value
fn unsigned_le_bytes(value: u128, width: usize) -> Vec<u8> {
    value.to_le_bytes()[..width].to_vec()
}

/// The low `width` bytes of a value, which is the cell a column of that
/// width stores
fn le_bytes(value: i128, width: usize) -> Vec<u8> {
    value.to_le_bytes()[..width].to_vec()
}

/// Whether a zone's bounds admit anything the term selects.
///
/// Slots hold raw little endian value bytes, so the comparison has to go
/// through the column's own signedness. A zone with no non-null value
/// records an inverted pair, which no interval overlaps, and that is
/// correct because a null satisfies no comparison
fn zone_admits(leaf: &Leaf, zone: &ZoneMapEntry) -> bool {
    let width = leaf.value_size;
    let cmp = |value: &[u8], slot: &[u8; STAT_VALUE_SIZE]| {
        compare_value_to_slot(value, slot, width, leaf.order)
    };
    match &leaf.admits {
        Admits::Values(values) => values.iter().any(|v| {
            cmp(v, &zone.min_value) != std::cmp::Ordering::Less
                && cmp(v, &zone.max_value) != std::cmp::Ordering::Greater
        }),
        Admits::Interval(low, high) => {
            let above = high
                .as_deref()
                .is_none_or(|hi| cmp(hi, &zone.min_value) != std::cmp::Ordering::Less);
            let below = low
                .as_deref()
                .is_none_or(|lo| cmp(lo, &zone.max_value) != std::cmp::Ordering::Greater);
            above && below
        }
    }
}

/// Reads whatever the file can answer without decoding it.
///
/// The caller supplies the column reads because this crate's file reader
/// owns them, and the two are kept apart so the lowering above stays
/// checkable against distributions rather than only against a file
pub(crate) trait ColumnEvidence {
    fn row_count(&self) -> usize;
    fn zone_maps(&self, column_id: u32) -> Result<Vec<ZoneMapEntry>, ZyronError>;
    fn eval(
        &self,
        column_id: u32,
        value_size: usize,
        predicate: &Predicate<'_>,
    ) -> Result<Vec<u8>, ZyronError>;
}

/// Rows the filter admits, as a keep bitmask of ceil(rows/8) bytes.
///
/// None means nothing was decided and every row stands. The mask is a
/// superset of the matching rows, never a subset, so the exact filter
/// still decides what is returned
pub(crate) fn rows_matching(
    filter: &StoredFilter,
    evidence: &dyn ColumnEvidence,
) -> Result<Option<Vec<u8>>, ZyronError> {
    eval_node(&filter.root, evidence)
}

fn eval_node(
    node: &StoredNode,
    evidence: &dyn ColumnEvidence,
) -> Result<Option<Vec<u8>>, ZyronError> {
    let rows = evidence.row_count();
    match node {
        StoredNode::All => Ok(None),
        StoredNode::Nothing => Ok(Some(vec![0u8; rows.div_ceil(8)])),
        StoredNode::Leaf(leaf) => eval_leaf(leaf, evidence),
        StoredNode::And(children) => {
            let mut mask: Option<Vec<u8>> = None;
            for child in children {
                let Some(child_mask) = eval_node(child, evidence)? else {
                    continue;
                };
                match &mut mask {
                    None => mask = Some(child_mask),
                    Some(acc) => {
                        for (a, b) in acc.iter_mut().zip(child_mask.iter()) {
                            *a &= *b;
                        }
                    }
                }
                // Nothing survives, so the remaining arms cost nothing
                if mask.as_ref().is_some_and(|m| m.iter().all(|b| *b == 0)) {
                    return Ok(mask);
                }
            }
            Ok(mask)
        }
        StoredNode::Or(children) => {
            let mut acc = vec![0u8; rows.div_ceil(8)];
            for child in children {
                // An arm that decides nothing admits everything, and so
                // does their union
                let Some(child_mask) = eval_node(child, evidence)? else {
                    return Ok(None);
                };
                for (a, b) in acc.iter_mut().zip(child_mask.iter()) {
                    *a |= *b;
                }
            }
            Ok(Some(acc))
        }
    }
}

/// Whether a zone holds nothing an inverted term selects.
///
/// Provable only when the zone's bounds pin every one of its rows to a
/// single value and that value is one the term excludes. A zone with no
/// non-null value records an inverted pair, which fails the equality and
/// is therefore kept, and keeping is always the safe direction
fn zone_excludes_everything(leaf: &Leaf, zone: &ZoneMapEntry) -> bool {
    let Admits::Values(values) = &leaf.admits else {
        return false;
    };
    if compare_stat_slots_typed(
        &zone.min_value,
        &zone.max_value,
        leaf.value_size,
        leaf.order,
    ) != std::cmp::Ordering::Equal
    {
        return false;
    }
    values.iter().any(|v| {
        compare_value_to_slot(v, &zone.min_value, leaf.value_size, leaf.order)
            == std::cmp::Ordering::Equal
    })
}

/// A term that selects what its values do not.
///
/// The mask is the equality mask inverted. Zone maps can only reject the
/// file when every zone is pinned to an excluded value, because one zone
/// that is not pinned is one zone with surviving rows
fn eval_inverted_leaf(
    leaf: &Leaf,
    evidence: &dyn ColumnEvidence,
) -> Result<Option<Vec<u8>>, ZyronError> {
    let rows = evidence.row_count();
    if leaf.zone_prunable {
        let zones = evidence.zone_maps(leaf.column_id)?;
        if !zones.is_empty() && zones.iter().all(|z| zone_excludes_everything(leaf, z)) {
            return Ok(Some(vec![0u8; rows.div_ceil(8)]));
        }
    }
    if leaf.pushdown.is_empty() {
        return Ok(None);
    }
    let mut mask = vec![0u8; rows.div_ceil(8)];
    for owned in &leaf.pushdown {
        let mut members: Vec<&[u8]> = Vec::new();
        let predicate = match owned {
            OwnedPredicate::AnyOf(values) => match values.as_slice() {
                [single] => Predicate::Equality(single),
                many => {
                    members.extend(many.iter().map(|v| v.as_slice()));
                    Predicate::In(&members)
                }
            },
            OwnedPredicate::Range { low, high } => Predicate::Range {
                low: low.as_deref(),
                high: high.as_deref(),
            },
        };
        let hit = evidence.eval(leaf.column_id, leaf.value_size, &predicate)?;
        for (a, b) in mask.iter_mut().zip(hit.iter()) {
            *a |= *b;
        }
    }
    for byte in mask.iter_mut() {
        *byte = !*byte;
    }
    // Bits past the last row are not rows, so they are cleared rather
    // than left as whatever the inversion made them
    if let Some(last) = mask.last_mut() {
        let used = rows % 8;
        if used != 0 {
            *last &= (1u8 << used) - 1;
        }
    }
    Ok(Some(mask))
}

fn eval_leaf(leaf: &Leaf, evidence: &dyn ColumnEvidence) -> Result<Option<Vec<u8>>, ZyronError> {
    let rows = evidence.row_count();
    if leaf.invert {
        return eval_inverted_leaf(leaf, evidence);
    }
    if leaf.zone_prunable {
        let zones = evidence.zone_maps(leaf.column_id)?;
        if !zones.is_empty() {
            let admitted: Vec<bool> = zones.iter().map(|z| zone_admits(leaf, z)).collect();
            if !admitted.iter().any(|a| *a) {
                // Not one zone can hold a matching row, so the payload is
                // never read
                return Ok(Some(vec![0u8; rows.div_ceil(8)]));
            }
            if leaf.pushdown.is_empty() {
                return Ok(Some(zone_mask(&admitted, rows)));
            }
        }
    }
    if leaf.pushdown.is_empty() {
        return Ok(None);
    }
    let mut acc = vec![0u8; rows.div_ceil(8)];
    for owned in &leaf.pushdown {
        // A membership of more than one value needs its members as
        // slices, which is the only shape that borrows anything beyond
        // the term itself
        let mut members: Vec<&[u8]> = Vec::new();
        let predicate = match owned {
            OwnedPredicate::AnyOf(values) => match values.as_slice() {
                [single] => Predicate::Equality(single),
                many => {
                    members.extend(many.iter().map(|v| v.as_slice()));
                    Predicate::In(&members)
                }
            },
            OwnedPredicate::Range { low, high } => Predicate::Range {
                low: low.as_deref(),
                high: high.as_deref(),
            },
        };
        let mask = evidence.eval(leaf.column_id, leaf.value_size, &predicate)?;
        for (a, b) in acc.iter_mut().zip(mask.iter()) {
            *a |= *b;
        }
    }
    Ok(Some(acc))
}

/// Expands a per-zone decision into a per-row keep mask
fn zone_mask(admitted: &[bool], rows: usize) -> Vec<u8> {
    let batch = ZONE_MAP_BATCH_SIZE as usize;
    let mut mask = vec![0u8; rows.div_ceil(8)];
    for (z, keep) in admitted.iter().enumerate() {
        if !keep {
            continue;
        }
        let start = z * batch;
        let end = ((z + 1) * batch).min(rows);
        for r in start..end {
            mask[r / 8] |= 1 << (r % 8);
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::LakeColumn;

    fn schema(types: &[TypeId]) -> LakeSchema {
        LakeSchema::new(
            1,
            types
                .iter()
                .enumerate()
                .map(|(i, t)| LakeColumn {
                    id: i as u32,
                    name: format!("c{}", i),
                    type_id: *t,
                    nullable: true,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                })
                .collect(),
        )
        .expect("valid schema")
    }

    fn cmp(column_id: u32, op: CompareOp, value: LakeValue) -> LakePredicate {
        LakePredicate::Compare {
            column_id,
            op,
            value,
        }
    }

    fn leaf_of(filter: &StoredFilter) -> &Leaf {
        match &filter.root {
            StoredNode::Leaf(leaf) => leaf,
            other => panic!("expected one leaf, got {:?}", other),
        }
    }

    /// A cell is admitted by a lowered range exactly when the value
    /// satisfies the original comparison, which is what makes the
    /// pushdown safe to hand an encoding
    #[test]
    fn test_a_signed_range_spanning_zero_becomes_two_stored_ranges() {
        let s = schema(&[TypeId::Int32]);
        let filter =
            StoredFilter::lower(&cmp(0, CompareOp::LtEq, LakeValue::Int(5)), &s).expect("lowers");
        let leaf = leaf_of(&filter);
        assert_eq!(leaf.pushdown.len(), 2, "negatives and non-negatives split");

        for v in [-2_000_000_000i32, -1, 0, 5, 6, 2_000_000_000] {
            let cell = v.to_le_bytes();
            let admitted = leaf.pushdown.iter().any(|p| match p {
                OwnedPredicate::Range { low, high } => {
                    zyron_storage::encoding::range_admits(&cell, 4, low.as_deref(), high.as_deref())
                }
                OwnedPredicate::AnyOf(_) => false,
            });
            assert_eq!(admitted, v <= 5, "value {} lowered wrong", v);
        }
    }

    #[test]
    fn test_an_unsigned_range_stays_one_stored_range() {
        let s = schema(&[TypeId::UInt32]);
        let filter =
            StoredFilter::lower(&cmp(0, CompareOp::Gt, LakeValue::UInt(100)), &s).expect("lowers");
        let leaf = leaf_of(&filter);
        assert_eq!(leaf.pushdown.len(), 1);
        for v in [0u32, 100, 101, u32::MAX] {
            let cell = v.to_le_bytes();
            let admitted = match &leaf.pushdown[0] {
                OwnedPredicate::Range { low, high } => {
                    zyron_storage::encoding::range_admits(&cell, 4, low.as_deref(), high.as_deref())
                }
                OwnedPredicate::AnyOf(_) => false,
            };
            assert_eq!(admitted, v > 100, "value {} lowered wrong", v);
        }
    }

    #[test]
    fn test_a_constant_outside_the_width_resolves_rather_than_wrapping() {
        let s = schema(&[TypeId::Int16]);
        // Every i16 is below this, so there is nothing left to lower
        assert!(
            StoredFilter::lower(&cmp(0, CompareOp::Lt, LakeValue::Int(1 << 40)), &s).is_none(),
            "a bound past the domain admits the whole column"
        );
        // And nothing is below the other end
        let filter = StoredFilter::lower(&cmp(0, CompareOp::Lt, LakeValue::Int(-(1 << 40))), &s)
            .expect("lowers");
        assert_eq!(filter.root, StoredNode::Nothing);
    }

    #[test]
    fn test_the_shapes_that_are_deliberately_not_lowered() {
        let s = schema(&[TypeId::Float64, TypeId::Int64, TypeId::Varchar]);
        // No comparison against NaN is ever true, so a NaN bound has no
        // position in the order to prune from
        assert!(
            StoredFilter::lower(&cmp(0, CompareOp::Lt, LakeValue::Float(f64::NAN)), &s).is_none()
        );
        // A null-shaped term is answered by the bitmap the decode returns
        assert!(StoredFilter::lower(&LakePredicate::IsNull { column_id: 1 }, &s).is_none());
        // NOT IN () excludes nothing
        assert!(
            StoredFilter::lower(
                &LakePredicate::Not(Box::new(LakePredicate::In {
                    column_id: 1,
                    values: vec![],
                })),
                &s
            )
            .is_none()
        );
    }

    /// A negated equality keeps what the constant does not pin. Its zone
    /// check runs the other way round from a positive term: one zone that
    /// is not pinned to an excluded value is one zone with rows that
    /// survive, so only an entirely pinned file can be rejected
    #[test]
    fn test_a_negated_equality_keeps_what_the_constant_does_not_pin() {
        let s = schema(&[TypeId::Int32]);
        let filter =
            StoredFilter::lower(&cmp(0, CompareOp::NotEq, LakeValue::Int(50)), &s).expect("lowers");
        let leaf = leaf_of(&filter);
        assert!(leaf.invert, "the mask is the equality mask inverted");
        assert_eq!(leaf.pushdown.len(), 1);
        match &leaf.pushdown[0] {
            OwnedPredicate::AnyOf(values) => {
                assert_eq!(values, &vec![50i32.to_le_bytes().to_vec()])
            }
            other => panic!("expected an equality, got {:?}", other),
        }

        let slot = |v: i32| {
            let mut s = [0u8; STAT_VALUE_SIZE];
            s[..4].copy_from_slice(&v.to_le_bytes());
            s
        };
        let zone = |min: i32, max: i32| ZoneMapEntry {
            min_value: slot(min),
            max_value: slot(max),
        };
        // Every row in this zone is 50, so none of them is `<> 50`
        assert!(zone_excludes_everything(leaf, &zone(50, 50)));
        // One value, but not the excluded one
        assert!(!zone_excludes_everything(leaf, &zone(7, 7)));
        // A spread of values holds something that survives
        assert!(!zone_excludes_everything(leaf, &zone(0, 100)));
        // A zone with no non-null value records an inverted pair and is kept
        assert!(!zone_excludes_everything(
            leaf,
            &ZoneMapEntry {
                min_value: [0xFF; STAT_VALUE_SIZE],
                max_value: [0u8; STAT_VALUE_SIZE],
            }
        ));

        // NOT IN carries every member it can prove
        let not_in = StoredFilter::lower(
            &LakePredicate::Not(Box::new(LakePredicate::In {
                column_id: 0,
                values: vec![LakeValue::Int(1), LakeValue::Int(2)],
            })),
            &s,
        )
        .expect("lowers");
        let not_in = leaf_of(&not_in);
        assert!(not_in.invert);
        assert!(zone_excludes_everything(not_in, &zone(2, 2)));
        assert!(!zone_excludes_everything(not_in, &zone(3, 3)));
    }

    #[test]
    fn test_a_disjunction_is_only_as_good_as_its_weakest_arm() {
        let s = schema(&[TypeId::Int64, TypeId::Int64]);
        // A null-shaped arm lowers to nothing, and one arm that admits
        // everything makes the union admit everything
        let mixed = LakePredicate::Or(vec![
            cmp(0, CompareOp::Lt, LakeValue::Int(10)),
            LakePredicate::IsNull { column_id: 1 },
        ]);
        assert!(StoredFilter::lower(&mixed, &s).is_none());

        // A conjunction keeps whatever its arms did lower
        let conj = LakePredicate::And(vec![
            cmp(0, CompareOp::Lt, LakeValue::Int(10)),
            LakePredicate::IsNull { column_id: 1 },
        ]);
        let filter = StoredFilter::lower(&conj, &s).expect("lowers the arm it can");
        assert_eq!(filter.columns(), vec![0]);
    }

    #[test]
    fn test_negation_is_pushed_down_to_the_leaves() {
        let s = schema(&[TypeId::Int64]);
        let direct =
            StoredFilter::lower(&cmp(0, CompareOp::GtEq, LakeValue::Int(10)), &s).expect("lowers");
        let negated = StoredFilter::lower(
            &LakePredicate::Not(Box::new(cmp(0, CompareOp::Lt, LakeValue::Int(10)))),
            &s,
        )
        .expect("lowers");
        assert_eq!(direct, negated, "NOT (x < 10) is x >= 10");

        // De Morgan turns the conjunction into a disjunction
        let and_not = StoredFilter::lower(
            &LakePredicate::Not(Box::new(LakePredicate::And(vec![
                cmp(0, CompareOp::GtEq, LakeValue::Int(0)),
                cmp(0, CompareOp::Lt, LakeValue::Int(10)),
            ]))),
            &s,
        )
        .expect("lowers");
        assert!(matches!(and_not.root, StoredNode::Or(_)));
    }

    #[test]
    fn test_a_zone_that_cannot_hold_a_match_is_rejected() {
        let s = schema(&[TypeId::Int32]);
        let filter =
            StoredFilter::lower(&cmp(0, CompareOp::Eq, LakeValue::Int(50)), &s).expect("lowers");
        let leaf = leaf_of(&filter);
        assert!(leaf.zone_prunable);

        let zone = |min: i32, max: i32| ZoneMapEntry {
            min_value: {
                let mut slot = [0u8; STAT_VALUE_SIZE];
                slot[..4].copy_from_slice(&min.to_le_bytes());
                slot
            },
            max_value: {
                let mut slot = [0u8; STAT_VALUE_SIZE];
                slot[..4].copy_from_slice(&max.to_le_bytes());
                slot
            },
        };
        assert!(zone_admits(leaf, &zone(0, 100)));
        assert!(!zone_admits(leaf, &zone(51, 100)));
        assert!(!zone_admits(leaf, &zone(-100, 49)));

        // A negative bound still sorts below a positive one
        let below =
            StoredFilter::lower(&cmp(0, CompareOp::Lt, LakeValue::Int(-5)), &s).expect("lowers");
        let below = leaf_of(&below);
        assert!(zone_admits(below, &zone(-100, -50)));
        assert!(!zone_admits(below, &zone(0, 100)));

        // A zone holding no non-null value records an inverted pair
        let all_null = ZoneMapEntry {
            min_value: [0xFF; STAT_VALUE_SIZE],
            max_value: [0u8; STAT_VALUE_SIZE],
        };
        assert!(!zone_admits(below, &all_null));
    }

    /// A sixteen-byte column ranges like any other. Its stored cell is
    /// the low sixteen bytes of the value, so a negative one reads as a
    /// very large unsigned number and the interval splits at zero
    #[test]
    fn test_a_128_bit_column_ranges_at_its_full_width() {
        let s = schema(&[TypeId::Int128, TypeId::UInt128]);

        let signed = StoredFilter::lower(&cmp(0, CompareOp::LtEq, LakeValue::Int128(5)), &s)
            .expect("lowers");
        let signed = leaf_of(&signed);
        assert_eq!(signed.value_size, 16);
        assert!(signed.zone_prunable, "a sixteen byte slot still compares");
        assert_eq!(signed.pushdown.len(), 2, "the interval spans zero");
        for v in [i128::MIN, -1i128, 0, 5, 6, i128::MAX] {
            let cell = v.to_le_bytes();
            let admitted = signed.pushdown.iter().any(|p| match p {
                OwnedPredicate::Range { low, high } => zyron_storage::encoding::range_admits(
                    &cell,
                    16,
                    low.as_deref(),
                    high.as_deref(),
                ),
                OwnedPredicate::AnyOf(_) => false,
            });
            assert_eq!(admitted, v <= 5, "signed value {} lowered wrong", v);
        }

        let unsigned = StoredFilter::lower(&cmp(1, CompareOp::Gt, LakeValue::UInt128(100)), &s)
            .expect("lowers");
        let unsigned = leaf_of(&unsigned);
        assert_eq!(unsigned.pushdown.len(), 1, "unsigned never spans zero");
        for v in [0u128, 100, 101, u128::MAX] {
            let cell = v.to_le_bytes();
            let admitted = match &unsigned.pushdown[0] {
                OwnedPredicate::Range { low, high } => zyron_storage::encoding::range_admits(
                    &cell,
                    16,
                    low.as_deref(),
                    high.as_deref(),
                ),
                OwnedPredicate::AnyOf(_) => false,
            };
            assert_eq!(admitted, v > 100, "unsigned value {} lowered wrong", v);
        }

        // The extremes resolve rather than overflowing the placement
        assert!(
            StoredFilter::lower(&cmp(0, CompareOp::Lt, LakeValue::Int128(i128::MIN)), &s)
                .is_some_and(|f| f.root == StoredNode::Nothing),
            "nothing is below the smallest value"
        );
        assert!(
            StoredFilter::lower(&cmp(0, CompareOp::LtEq, LakeValue::Int128(i128::MAX)), &s)
                .is_none(),
            "everything is at or below the largest value"
        );
        assert!(
            StoredFilter::lower(&cmp(1, CompareOp::GtEq, LakeValue::Int(-1)), &s).is_none(),
            "every unsigned value is at or above a negative bound"
        );
        assert!(
            StoredFilter::lower(&cmp(1, CompareOp::Lt, LakeValue::UInt(0)), &s)
                .is_some_and(|f| f.root == StoredNode::Nothing),
            "no unsigned value is below zero"
        );
    }

    /// Stat slots order floats by value now, so a float range rejects
    /// zones the way any other range does. It is not pushed into the
    /// encoding, and equality is, because equality is byte equality and
    /// every encoding agrees on that
    #[test]
    fn test_a_float_column_prunes_zones_by_value() {
        let s = schema(&[TypeId::Float64]);
        let slot = |v: f64| {
            let mut out = [0u8; STAT_VALUE_SIZE];
            out[..8].copy_from_slice(&v.to_le_bytes());
            out
        };
        let zone = |min: f64, max: f64| ZoneMapEntry {
            min_value: slot(min),
            max_value: slot(max),
        };

        let below = StoredFilter::lower(&cmp(0, CompareOp::Lt, LakeValue::Float(-1.0)), &s)
            .expect("lowers");
        let below = leaf_of(&below);
        assert!(below.zone_prunable);
        assert!(
            below.pushdown.is_empty(),
            "a float range stops at the zone maps"
        );
        // A negative bound sorts below every positive, which an unsigned
        // reading of the same bytes would have got backwards
        assert!(zone_admits(below, &zone(-100.0, -50.0)));
        assert!(!zone_admits(below, &zone(0.0, 100.0)));
        assert!(!zone_admits(below, &zone(-0.5, 3.0)));
        assert!(zone_admits(below, &zone(-2.0, 5.0)));

        // Equality is pushed, and zero admits both of its spellings
        let zero =
            StoredFilter::lower(&cmp(0, CompareOp::Eq, LakeValue::Float(0.0)), &s).expect("lowers");
        let zero = leaf_of(&zero);
        match &zero.pushdown[0] {
            OwnedPredicate::AnyOf(values) => assert_eq!(
                values.len(),
                2,
                "negative zero equals zero and has different bytes"
            ),
            other => panic!("expected an equality set, got {:?}", other),
        }
        assert!(zone_admits(zero, &zone(-1.0, 1.0)));
        assert!(!zone_admits(zero, &zone(1.0, 2.0)));

        // A non-zero constant carries one spelling
        let one =
            StoredFilter::lower(&cmp(0, CompareOp::Eq, LakeValue::Float(1.5)), &s).expect("lowers");
        match &leaf_of(&one).pushdown[0] {
            OwnedPredicate::AnyOf(values) => assert_eq!(values.len(), 1),
            other => panic!("expected an equality set, got {:?}", other),
        }
    }

    #[test]
    fn test_a_variable_length_range_keeps_an_inclusive_bound() {
        let s = schema(&[TypeId::Varchar]);
        let filter = StoredFilter::lower(&cmp(0, CompareOp::Lt, LakeValue::Str("m".into())), &s)
            .expect("lowers");
        let leaf = leaf_of(&filter);
        assert!(
            leaf.zone_prunable,
            "varlen slots hold a lexicographic prefix"
        );
        match &leaf.pushdown[0] {
            OwnedPredicate::Range { low, high } => {
                assert!(low.is_none());
                assert_eq!(high.as_deref(), Some(&b"m"[..]));
            }
            other => panic!("expected a range, got {:?}", other),
        }
    }
}
