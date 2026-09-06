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
//! A null cell is stored as zero bytes, so a comparison answered on the
//! payload sees the value zero where the row holds nothing. Every leaf
//! that reads the payload drops the null rows from its own mask, using the
//! bitmap that came off the same read, and that applies to a term and to
//! its inverse alike: SQL says a null satisfies no comparison on either
//! side of a negation. `IS NULL` and `IS NOT NULL` are that bitmap and
//! nothing else.
//!
//! Everything here produces a superset of the matching rows and never a
//! subset, and [`StoredFilter::is_exact`] says when the superset is the
//! set. Exact means every term lowered, every one of them answers on the
//! payload rather than on zone bounds alone, and no bound was widened to
//! reach a form the encodings share. A caller holding an exact mask has
//! nothing left to check, so the row filter over decoded values, and the
//! columns projected only to feed it, are work the scan has already done.
//!
//! What is not exact, and why:
//!
//! * a float range. Stat slots order floats correctly, so it still rejects
//!   zones, but `Predicate::Range` is defined over unsigned byte order and
//!   ALP answers one in float order, so the encodings do not agree on it
//!   and nothing is pushed. Float equality is byte equality and every
//!   encoding agrees on that, so it is pushed and is exact away from NaN
//! * `<` on a variable-length column. A byte string has no greatest
//!   predecessor, so the bound stays inclusive and the rows equal to the
//!   constant are left to the row filter. `>` has an immediate successor,
//!   the constant with a zero byte appended, and is exact
//! * a term against a NaN, which no comparison is ever true of, though two
//!   NaNs with the same bits are the same cell
//!
//! Nothing else is held back. A variable-length column prunes zones from
//! the prefix its slots hold, which is a bound in both directions, and the
//! payload evaluation behind it reads whole values.
//!
//! `<>` and `NOT IN` are lowered as the equality mask inverted, with the
//! null rows removed from the result rather than left on the keep side.

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
    exact: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum StoredNode {
    /// Nothing was lowered, so every row stands
    All,
    /// The term is provably empty
    Nothing,
    Leaf(Leaf),
    /// Rows whose cell is null, or the rows whose cell is not, read from
    /// the segment's own null bitmap with no payload decoded
    Null {
        column_id: u32,
        keep_null: bool,
    },
    And(Vec<StoredNode>),
    Or(Vec<StoredNode>),
}

/// One lowered term and whether its mask is the matching rows themselves.
///
/// Exactness is decided here rather than read back off the tree, because a
/// conjunction drops the arms that lowered to nothing and the tree it
/// leaves cannot say an arm was ever there
struct Lowered {
    node: StoredNode,
    /// The mask selects the matching rows and not a superset of them
    exact: bool,
}

impl Lowered {
    /// Nothing was lowered, so every row stands and the term is unanswered
    fn all() -> Self {
        Self {
            node: StoredNode::All,
            exact: false,
        }
    }

    /// The term is provably empty, which is as exact as an answer gets
    fn nothing() -> Self {
        Self {
            node: StoredNode::Nothing,
            exact: true,
        }
    }

    fn leaf(leaf: Leaf, exact: bool) -> Self {
        Self {
            node: StoredNode::Leaf(leaf),
            exact,
        }
    }

    /// Every value the column can hold satisfies the term, so what is left
    /// of it is that a null satisfies no comparison
    fn not_null(column_id: u32) -> Self {
        Self {
            node: StoredNode::Null {
                column_id,
                keep_null: false,
            },
            exact: true,
        }
    }
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
        let lowered = lower_node(predicate, schema, false);
        match lowered.node {
            StoredNode::All => None,
            root => Some(Self {
                root,
                exact: lowered.exact,
            }),
        }
    }

    /// Whether the mask this produces is the matching rows themselves
    /// rather than a superset of them.
    ///
    /// Exact means every term lowered, every one of them answers on the
    /// payload rather than on zone bounds alone, and each leaf drops the
    /// null rows its comparison cannot match. A caller that holds an exact
    /// mask has nothing left to check, so the row filter over decoded
    /// values, and the columns that exist only to feed it, are work the
    /// scan has already done
    pub fn is_exact(&self) -> bool {
        self.exact
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
        StoredNode::Null { column_id, .. } => ids.push(*column_id),
        StoredNode::And(children) | StoredNode::Or(children) => {
            for c in children {
                collect_columns(c, ids);
            }
        }
        StoredNode::All | StoredNode::Nothing => {}
    }
}

fn lower_node(predicate: &LakePredicate, schema: &LakeSchema, negated: bool) -> Lowered {
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
        // A segment records its rows' nullness in a bitmap ahead of its
        // payload, so a null-shaped term is answered by reading that
        // bitmap and nothing else. It is the whole answer rather than a
        // bound on it, whatever the column holds
        LakePredicate::IsNull { column_id } | LakePredicate::IsNotNull { column_id } => {
            let wants_null = matches!(predicate, LakePredicate::IsNull { .. }) != negated;
            if schema.column_by_id(*column_id).is_none() {
                return Lowered::all();
            }
            Lowered {
                node: StoredNode::Null {
                    column_id: *column_id,
                    keep_null: wants_null,
                },
                exact: true,
            }
        }
        LakePredicate::In { column_id, values } if !negated => {
            if values.is_empty() {
                return Lowered::nothing();
            }
            let Some(shape) = column_shape(*column_id, schema) else {
                return Lowered::all();
            };
            let mut cells = Vec::with_capacity(values.len());
            for value in values {
                match value_to_cell(shape.physical, shape.value_size, value) {
                    // One member with no provable stored form makes the
                    // whole membership unprovable, since the rows it would
                    // have admitted must not be dropped
                    None => return Lowered::all(),
                    Some(cell) => cells.push(cell.as_slice().to_vec()),
                }
            }
            let exact = values.iter().all(|v| equality_is_exact(&shape, v));
            Lowered::leaf(shape.equality_leaf(cells), exact)
        }
        // NOT IN is the membership mask inverted. A member with no
        // provable stored form is skipped, which only widens the result
        LakePredicate::In { column_id, values } => {
            let Some(shape) = column_shape(*column_id, schema) else {
                return Lowered::all();
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
                return Lowered::all();
            }
            // A dropped member leaves rows the term excludes on the keep
            // side, so the mask is a bound rather than the answer
            let exact =
                cells.len() == values.len() && values.iter().all(|v| equality_is_exact(&shape, v));
            Lowered::leaf(shape.inverted_leaf(cells), exact)
        }
        LakePredicate::And(children) | LakePredicate::Or(children) => {
            let conjunction = matches!(predicate, LakePredicate::And(_)) != negated;
            let lowered: Vec<Lowered> = children
                .iter()
                .map(|c| lower_node(c, schema, negated))
                .collect();
            // Every arm has to answer its own term for the whole to answer
            // the predicate, in both directions: a dropped conjunct leaves
            // rows it would have removed, and a widened disjunct is already
            // wider than the union
            let exact = lowered.iter().all(|l| l.exact);
            if conjunction {
                // An arm that lowered to nothing is dropped, which keeps
                // the result a superset
                if lowered.iter().any(|l| l.node == StoredNode::Nothing) {
                    return Lowered::nothing();
                }
                let kept: Vec<StoredNode> = lowered
                    .into_iter()
                    .map(|l| l.node)
                    .filter(|n| *n != StoredNode::All)
                    .collect();
                match kept.len() {
                    0 => Lowered::all(),
                    1 => match kept.into_iter().next() {
                        Some(node) => Lowered { node, exact },
                        None => Lowered::all(),
                    },
                    _ => Lowered {
                        node: StoredNode::And(kept),
                        exact,
                    },
                }
            } else {
                // A disjunction is only as good as its weakest arm: one
                // arm that admits everything makes the whole term admit
                // everything
                if lowered.iter().any(|l| l.node == StoredNode::All) {
                    return Lowered::all();
                }
                let kept: Vec<StoredNode> = lowered
                    .into_iter()
                    .map(|l| l.node)
                    .filter(|n| *n != StoredNode::Nothing)
                    .collect();
                match kept.len() {
                    0 => Lowered::nothing(),
                    1 => match kept.into_iter().next() {
                        Some(node) => Lowered { node, exact },
                        None => Lowered::all(),
                    },
                    _ => Lowered {
                        node: StoredNode::Or(kept),
                        exact,
                    },
                }
            }
        }
    }
}

/// Whether byte equality against this constant is the comparison SQL
/// defines.
///
/// It is, for every value with a stored form, except a NaN. Two NaNs with
/// the same bits are the same cell and compare unequal all the same, so a
/// term against one is answered by the row filter rather than here
fn equality_is_exact(shape: &ColumnShape, value: &LakeValue) -> bool {
    !(shape.float && matches!(value, LakeValue::Float(v) if v.is_nan()))
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

fn lower_compare(column_id: u32, op: CompareOp, value: &LakeValue, schema: &LakeSchema) -> Lowered {
    let Some(shape) = column_shape(column_id, schema) else {
        return Lowered::all();
    };
    match op {
        CompareOp::Eq => match equality_cells(&shape, value) {
            Some(cells) => {
                Lowered::leaf(shape.equality_leaf(cells), equality_is_exact(&shape, value))
            }
            None => Lowered::all(),
        },
        // The mask is the equality mask inverted, which keeps every row
        // the constant does not pin
        CompareOp::NotEq => match equality_cells(&shape, value) {
            Some(cells) => {
                Lowered::leaf(shape.inverted_leaf(cells), equality_is_exact(&shape, value))
            }
            None => Lowered::all(),
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
fn lower_float_range(shape: &ColumnShape, op: CompareOp, value: &LakeValue) -> Lowered {
    if matches!(value, LakeValue::Float(v) if v.is_nan()) {
        return Lowered::all();
    }
    let Some(cell) = value_to_cell(shape.physical, shape.value_size, value) else {
        return Lowered::all();
    };
    let bound = cell.as_slice().to_vec();
    let (low, high) = match op {
        CompareOp::Lt | CompareOp::LtEq => (None, Some(bound)),
        CompareOp::Gt | CompareOp::GtEq => (Some(bound), None),
        CompareOp::Eq | CompareOp::NotEq => return Lowered::all(),
    };
    if !shape.zone_prunable {
        return Lowered::all();
    }
    Lowered::leaf(
        Leaf {
            column_id: shape.column_id,
            value_size: shape.value_size,
            order: shape.order,
            zone_prunable: true,
            admits: Admits::Interval(low, high),
            pushdown: Vec::new(),
            invert: false,
        },
        false,
    )
}

/// A range over a variable-length column, whose stored bytes are ordered
/// lexicographically by the same rule its values are.
///
/// `>` takes the constant's immediate successor, which for a byte string
/// is the constant with a zero byte appended: nothing sorts between the
/// two, and every string above the constant is at or above it. That makes
/// a strict lower bound the answer rather than a bound on it.
///
/// `<` has no such form. A byte string has no greatest predecessor,
/// because appending 0xff bytes to any candidate produces another one
/// still below the constant, so the bound stays inclusive there and the
/// rows equal to the constant are left to the exact filter.
///
/// Zone bounds keep the inclusive constant either way. Slots hold a
/// 32-byte prefix, and lengthening the value being compared against a
/// truncated prefix is not a direction the comparison stays a bound in
fn lower_varlen_range(shape: &ColumnShape, op: CompareOp, value: &LakeValue) -> Lowered {
    let Some(cell) = value_to_cell(shape.physical, shape.value_size, value) else {
        return Lowered::all();
    };
    let bound = cell.as_slice().to_vec();
    let (low, high) = match op {
        CompareOp::Lt | CompareOp::LtEq => (None, Some(bound.clone())),
        CompareOp::Gt | CompareOp::GtEq => (Some(bound.clone()), None),
        CompareOp::Eq | CompareOp::NotEq => return Lowered::all(),
    };
    let (pushdown_low, pushdown_high, exact) = match op {
        CompareOp::Gt => {
            let mut successor = bound;
            successor.push(0);
            (Some(successor), None, true)
        }
        CompareOp::GtEq => (Some(bound), None, true),
        CompareOp::LtEq => (None, Some(bound), true),
        CompareOp::Lt => (None, Some(bound), false),
        CompareOp::Eq | CompareOp::NotEq => return Lowered::all(),
    };
    Lowered::leaf(
        Leaf {
            column_id: shape.column_id,
            value_size: 0,
            order: shape.order,
            zone_prunable: shape.zone_prunable,
            admits: Admits::Interval(low, high),
            pushdown: vec![OwnedPredicate::Range {
                low: pushdown_low,
                high: pushdown_high,
            }],
            invert: false,
        },
        exact,
    )
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
fn lower_numeric_range(shape: &ColumnShape, op: CompareOp, value: &LakeValue) -> Lowered {
    let width = shape.value_size;
    if !(1..=MAX_RANGE_WIDTH).contains(&width) {
        return Lowered::all();
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
) -> Lowered {
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
        _ => return Lowered::all(),
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
        (CompareOp::Eq | CompareOp::NotEq, _) => return Lowered::all(),
    };
    let Some((lo, hi)) = interval else {
        return Lowered::nothing();
    };
    if lo == 0 && hi == umax {
        return Lowered::not_null(shape.column_id);
    }

    let low = (lo > 0).then(|| unsigned_le_bytes(lo, width));
    let high = (hi < umax).then(|| unsigned_le_bytes(hi, width));
    Lowered::leaf(
        Leaf {
            column_id: shape.column_id,
            value_size: width,
            order: shape.order,
            zone_prunable: shape.zone_prunable,
            admits: Admits::Interval(low.clone(), high.clone()),
            pushdown: vec![OwnedPredicate::Range { low, high }],
            invert: false,
        },
        true,
    )
}

fn lower_signed_range(
    shape: &ColumnShape,
    width: usize,
    op: CompareOp,
    value: &LakeValue,
) -> Lowered {
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
        _ => return Lowered::all(),
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
        (CompareOp::Eq | CompareOp::NotEq, _) => return Lowered::all(),
    };
    let Some((lo, hi)) = interval else {
        return Lowered::nothing();
    };
    if lo == smin && hi == smax {
        return Lowered::not_null(shape.column_id);
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

    Lowered::leaf(
        Leaf {
            column_id: shape.column_id,
            value_size: width,
            order: shape.order,
            zone_prunable: shape.zone_prunable,
            admits: Admits::Interval(bounded_low, bounded_high),
            pushdown,
            invert: false,
        },
        true,
    )
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
    fn zone_maps(&self, column_id: u32) -> Result<std::sync::Arc<[ZoneMapEntry]>, ZyronError>;
    /// Which of a column's rows hold a value.
    ///
    /// A segment writes its null bitmap immediately ahead of its payload
    /// and only when it has a null at all, so this is answered from the
    /// segment header for a column with no nulls and from bytes the
    /// payload read already pulled for one with them
    fn validity(&self, column_id: u32) -> Result<Validity, ZyronError>;
    /// Whether the column's value bloom proves it holds none of `values`.
    ///
    /// False when the filter admits one of them and when the segment
    /// carries no filter at all, which are the same answer here: nothing
    /// was proven. A segment that carries one is a segment whose values are
    /// spread widely enough that its bounds narrow nothing, which is
    /// exactly where an equality otherwise reads the whole payload
    fn bloom_denies(&self, column_id: u32, values: &[Vec<u8>]) -> Result<bool, ZyronError>;
    /// Rows `start..end` the predicate admits, as a keep mask covering that
    /// range alone, `ceil((end - start) / 8)` bytes.
    ///
    /// The range is what zone pruning left standing. Answering for the whole
    /// column when one zone survived does the work of every row in the file
    /// to describe a thousand of them
    fn eval(
        &self,
        column_id: u32,
        value_size: usize,
        predicate: &Predicate<'_>,
        start: usize,
        end: usize,
    ) -> Result<Vec<u8>, ZyronError>;
}

/// Which of a column's rows hold a value.
///
/// A column the file predates has no segment at all, so every one of its
/// rows reads as null and no comparison over it matches
pub(crate) enum Validity {
    /// Every row holds a value, so nothing is removed for nullness
    AllValid,
    /// Every row is null
    AllNull,
    /// Set bit means the row is null, one bit per row of the whole file
    Nulls(Vec<u8>),
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

/// Removes the rows whose cell is null from a mask.
///
/// Every comparison SQL defines is false for a null operand, on both
/// sides of a negation, so this applies to a term and to its inverse
/// alike. It runs on a mask a payload evaluation already produced, where
/// the null bitmap came off the same read as the payload
fn apply_validity(mask: &mut [u8], validity: &Validity, rows: usize) {
    match validity {
        Validity::AllValid => {}
        Validity::AllNull => mask.fill(0),
        Validity::Nulls(nulls) => {
            for (m, n) in mask.iter_mut().zip(nulls.iter()) {
                *m &= !*n;
            }
            // A bitmap shorter than the mask leaves rows it says nothing
            // about, and a row with no recorded nullness is not a row
            if nulls.len() < mask.len() {
                for m in mask[nulls.len()..].iter_mut() {
                    *m = 0;
                }
            }
        }
    }
    clear_tail_bits(mask, rows);
}

/// Clears the bits past the last row, which are not rows
fn clear_tail_bits(mask: &mut [u8], rows: usize) {
    if let Some(last) = mask.last_mut() {
        let used = rows % 8;
        if used != 0 {
            *last &= (1u8 << used) - 1;
        }
    }
}

/// The rows a null-shaped term selects, read from the null bitmap alone.
///
/// None means every row stands, which is what a column with no null has
/// to say about IS NOT NULL
fn eval_null_node(
    column_id: u32,
    keep_null: bool,
    evidence: &dyn ColumnEvidence,
) -> Result<Option<Vec<u8>>, ZyronError> {
    let rows = evidence.row_count();
    let validity = evidence.validity(column_id)?;
    Ok(match (validity, keep_null) {
        (Validity::AllValid, true) | (Validity::AllNull, false) => {
            Some(vec![0u8; rows.div_ceil(8)])
        }
        (Validity::AllValid, false) | (Validity::AllNull, true) => None,
        (Validity::Nulls(nulls), true) => {
            let mut mask = vec![0u8; rows.div_ceil(8)];
            let shared = nulls.len().min(mask.len());
            mask[..shared].copy_from_slice(&nulls[..shared]);
            clear_tail_bits(&mut mask, rows);
            Some(mask)
        }
        (Validity::Nulls(nulls), false) => {
            let mut mask = vec![0xffu8; rows.div_ceil(8)];
            apply_validity(&mut mask, &Validity::Nulls(nulls), rows);
            Some(mask)
        }
    })
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
        StoredNode::Null {
            column_id,
            keep_null,
        } => eval_null_node(*column_id, *keep_null, evidence),
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
        // Every row, not the admitted zones. An inverted leaf keeps exactly
        // the rows the positive predicate rejects, and a zone the positive
        // predicate admits still holds rows it does not match, so narrowing
        // to admitted zones here would drop the survivors everywhere else
        let hit = evidence.eval(leaf.column_id, leaf.value_size, &predicate, 0, rows)?;
        for (a, b) in mask.iter_mut().zip(hit.iter()) {
            *a |= *b;
        }
    }
    for byte in mask.iter_mut() {
        *byte = !*byte;
    }
    // A null cell is zero filled, so it fails the equality and lands on
    // the keep side of the inversion. `x <> c` is false for a null x the
    // same way `x = c` is, so the same removal applies here. Bits past the
    // last row are cleared with it rather than left as the inversion made
    // them
    apply_validity(&mut mask, &evidence.validity(leaf.column_id)?, rows);
    Ok(Some(mask))
}

fn eval_leaf(leaf: &Leaf, evidence: &dyn ColumnEvidence) -> Result<Option<Vec<u8>>, ZyronError> {
    let rows = evidence.row_count();
    if leaf.invert {
        return eval_inverted_leaf(leaf, evidence);
    }
    // Row spans the zone maps could not reject, which is all the payload
    // evaluation has to answer for. Whole file when nothing pruned
    let mut spans: Vec<(usize, usize)> = Vec::new();
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
            spans = admitted_spans(&admitted, rows);
        }
    }
    if leaf.pushdown.is_empty() {
        return Ok(None);
    }
    // Zone bounds only say a constant falls inside a range they cover, and
    // a column whose values are spread across its range has a zone covering
    // every constant. The value bloom answers whether the segment holds the
    // constant at all, and it is built for exactly those columns, so an
    // equality that no row satisfies stops before the payload is read
    if let Admits::Values(values) = &leaf.admits
        && evidence.bloom_denies(leaf.column_id, values)?
    {
        return Ok(Some(vec![0u8; rows.div_ceil(8)]));
    }
    if spans.is_empty() {
        spans.push((0, rows));
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
        for &(start, end) in &spans {
            let mask = evidence.eval(leaf.column_id, leaf.value_size, &predicate, start, end)?;
            // A zone is a whole number of bytes of mask, so a span starts on
            // a byte boundary and its answer drops in without shifting
            let byte_start = start / 8;
            for (offset, byte) in mask.iter().enumerate() {
                if let Some(slot) = acc.get_mut(byte_start + offset) {
                    *slot |= *byte;
                }
            }
        }
    }
    // A null cell is zero filled and compares as the value zero would, so
    // it can pass a range or an equality the row does not satisfy. The
    // bitmap that says so came off the same read as the payload
    apply_validity(&mut acc, &evidence.validity(leaf.column_id)?, rows);
    Ok(Some(acc))
}

/// Contiguous row spans the admitted zones cover.
///
/// Adjacent zones are merged so a clustered column, where the matching rows
/// sit together, evaluates one span rather than one call per zone. Zone
/// width divides eight, so every span starts on a mask byte boundary and
/// its answer needs no bit shifting to merge
fn admitted_spans(admitted: &[bool], rows: usize) -> Vec<(usize, usize)> {
    let batch = ZONE_MAP_BATCH_SIZE as usize;
    let mut spans = Vec::new();
    let mut open: Option<usize> = None;
    for (zone, keep) in admitted.iter().enumerate() {
        match (keep, open) {
            (true, None) => open = Some(zone * batch),
            (false, Some(start)) => {
                spans.push((start, (zone * batch).min(rows)));
                open = None;
            }
            _ => {}
        }
    }
    if let Some(start) = open {
        spans.push((start, rows));
    }
    spans
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
        // Every i16 is below this, so what is left of the term is that a
        // null satisfies no comparison
        let filter = StoredFilter::lower(&cmp(0, CompareOp::Lt, LakeValue::Int(1 << 40)), &s)
            .expect("a bound past the domain still excludes the nulls");
        assert_eq!(
            filter.root,
            StoredNode::Null {
                column_id: 0,
                keep_null: false
            }
        );
        assert!(filter.is_exact());
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
        let s = schema(&[TypeId::Int64, TypeId::Float64]);
        // A NaN bound has no position in the order, so that arm admits
        // everything and so does the union
        let mixed = LakePredicate::Or(vec![
            cmp(0, CompareOp::Lt, LakeValue::Int(10)),
            cmp(1, CompareOp::Lt, LakeValue::Float(f64::NAN)),
        ]);
        assert!(StoredFilter::lower(&mixed, &s).is_none());

        // A conjunction keeps whatever its arms did lower, and reports
        // itself short of exact because the dropped arm still selects
        let conj = LakePredicate::And(vec![
            cmp(0, CompareOp::Lt, LakeValue::Int(10)),
            cmp(1, CompareOp::Lt, LakeValue::Float(f64::NAN)),
        ]);
        let filter = StoredFilter::lower(&conj, &s).expect("lowers the arm it can");
        assert_eq!(filter.columns(), vec![0]);
        assert!(!filter.is_exact());
    }

    /// A null-shaped term reads the segment's own null bitmap, which is the
    /// whole answer rather than a bound on it
    #[test]
    fn test_a_null_shaped_term_lowers_onto_the_null_bitmap() {
        let s = schema(&[TypeId::Int64]);
        let is_null = StoredFilter::lower(&LakePredicate::IsNull { column_id: 0 }, &s)
            .expect("IS NULL lowers");
        assert_eq!(
            is_null.root,
            StoredNode::Null {
                column_id: 0,
                keep_null: true
            }
        );
        assert!(is_null.is_exact());

        let not_null = StoredFilter::lower(&LakePredicate::IsNotNull { column_id: 0 }, &s)
            .expect("IS NOT NULL lowers");
        assert_eq!(
            not_null.root,
            StoredNode::Null {
                column_id: 0,
                keep_null: false
            }
        );

        // NOT wrapping either one swaps which side it keeps
        let negated = StoredFilter::lower(
            &LakePredicate::Not(Box::new(LakePredicate::IsNull { column_id: 0 })),
            &s,
        )
        .expect("NOT IS NULL lowers");
        assert_eq!(negated.root, not_null.root);

        // A column the schema does not name has no bitmap to read
        assert!(StoredFilter::lower(&LakePredicate::IsNull { column_id: 7 }, &s).is_none());
    }

    /// Exactness is a property of every term at once. One arm that only
    /// bounds its rows makes the whole mask a bound
    #[test]
    fn test_exactness_holds_only_when_every_term_answers_its_own_rows() {
        let s = schema(&[TypeId::Int64, TypeId::Float64, TypeId::Varchar]);
        assert!(
            StoredFilter::lower(&cmp(0, CompareOp::Gt, LakeValue::Int(10)), &s)
                .expect("lowers")
                .is_exact(),
            "an integer range answers on the payload"
        );
        // A float range prunes zones and pushes nothing, so it bounds
        assert!(
            !StoredFilter::lower(&cmp(1, CompareOp::Gt, LakeValue::Float(1.5)), &s)
                .expect("lowers")
                .is_exact()
        );
        // A strict lower bound on a byte string has an immediate successor
        assert!(
            StoredFilter::lower(
                &cmp(2, CompareOp::Gt, LakeValue::Str("abc".to_string())),
                &s
            )
            .expect("lowers")
            .is_exact()
        );
        // A strict upper bound has no greatest predecessor, so the rows
        // equal to the constant stay in
        assert!(
            !StoredFilter::lower(
                &cmp(2, CompareOp::Lt, LakeValue::Str("abc".to_string())),
                &s
            )
            .expect("lowers")
            .is_exact()
        );
        // One inexact arm carries through a conjunction
        let conj = LakePredicate::And(vec![
            cmp(0, CompareOp::Gt, LakeValue::Int(10)),
            cmp(1, CompareOp::Gt, LakeValue::Float(1.5)),
        ]);
        assert!(!StoredFilter::lower(&conj, &s).expect("lowers").is_exact());
    }

    /// A strict lower bound on a byte string admits what is above the
    /// constant and not the constant itself
    #[test]
    fn test_a_strict_string_lower_bound_excludes_the_constant() {
        let s = schema(&[TypeId::Varchar]);
        let filter = StoredFilter::lower(
            &cmp(0, CompareOp::Gt, LakeValue::Str("abc".to_string())),
            &s,
        )
        .expect("lowers");
        let leaf = leaf_of(&filter);
        let OwnedPredicate::Range { low, high } = &leaf.pushdown[0] else {
            panic!("a range lowers to a range");
        };
        assert_eq!(low.as_deref(), Some(b"abc\0".as_slice()));
        assert_eq!(high.as_deref(), None);
        for (value, expected) in [
            (b"abc".as_slice(), false),
            (b"abc\0".as_slice(), true),
            (b"abcd".as_slice(), true),
            (b"abd".as_slice(), true),
            (b"abb".as_slice(), false),
        ] {
            assert_eq!(
                zyron_storage::encoding::range_admits(value, 0, low.as_deref(), high.as_deref()),
                expected,
                "{:?} lowered wrong",
                value
            );
        }
        // Zone bounds keep the inclusive constant, because a slot holds a
        // truncated prefix and a longer value is not a bound against one
        let Admits::Interval(zone_low, _) = &leaf.admits else {
            panic!("a range admits an interval");
        };
        assert_eq!(zone_low.as_deref(), Some(b"abc".as_slice()));
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
        let all_of_them = StoredNode::Null {
            column_id: 0,
            keep_null: false,
        };
        assert!(
            StoredFilter::lower(&cmp(0, CompareOp::LtEq, LakeValue::Int128(i128::MAX)), &s)
                .is_some_and(|f| f.root == all_of_them),
            "everything is at or below the largest value, except a null"
        );
        assert!(
            StoredFilter::lower(&cmp(1, CompareOp::GtEq, LakeValue::Int(-1)), &s).is_some_and(
                |f| f.root
                    == StoredNode::Null {
                        column_id: 1,
                        keep_null: false
                    }
            ),
            "every unsigned value is at or above a negative bound, except a null"
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
