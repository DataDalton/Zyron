//! Typed predicate IR for pruning and predicate-based deletes.
//!
//! A predicate here is typed and column-id addressed so it can be compared
//! against per-file bounds without SQL machinery. Evaluation against file
//! statistics is three-valued: a file the predicate cannot match is skipped,
//! a file it fully covers can be dropped outright by a delete, and anything
//! else is scanned. Every decision errs toward MayMatch, never away from it.
//!
//! Null semantics follow SQL. A comparison over a null row is unknown and
//! selects nothing, so FullyCovers for comparisons additionally requires a
//! null-free column. Negation is pushed down exactly, `NOT (a = b)` and
//! `a <> b` have identical three-valued truth tables

use std::cmp::Ordering;

use zyron_common::ZyronError;

use crate::codec::Cursor;

/// Typed constant in a predicate. The planner lowers each bound constant
/// into the column's value family, cross-family comparisons outside the
/// integer widenings resolve to unknown
#[derive(Debug, Clone, PartialEq)]
pub enum LakeValue {
    Null,
    Bool(bool),
    /// All signed integers up to 64 bits, dates, times and timestamps
    Int(i64),
    /// 128-bit integers, picosecond timestamps, HLC, scaled decimals
    Int128(i128),
    /// Unsigned integers up to 64 bits
    UInt(u64),
    UInt128(u128),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
}

impl LakeValue {
    /// Exact ordering between two values, None when the pair is not
    /// comparable. Integer variants widen and compare exactly across
    /// signedness, floats use IEEE total order
    pub fn compare(&self, other: &LakeValue) -> Option<Ordering> {
        use LakeValue::*;
        match (self, other) {
            (Bool(a), Bool(b)) => Some(a.cmp(b)),
            (Float(a), Float(b)) => Some(a.total_cmp(b)),
            (Str(a), Str(b)) => Some(a.as_bytes().cmp(b.as_bytes())),
            (Bytes(a), Bytes(b)) => Some(a.cmp(b)),
            (
                Int(_) | Int128(_) | UInt(_) | UInt128(_),
                Int(_) | Int128(_) | UInt(_) | UInt128(_),
            ) => Some(compare_integers(self.as_num(), other.as_num())),
            _ => None,
        }
    }

    fn as_num(&self) -> Num {
        match self {
            LakeValue::Int(v) => Num::I(*v as i128),
            LakeValue::Int128(v) => Num::I(*v),
            LakeValue::UInt(v) => Num::U(*v as u128),
            LakeValue::UInt128(v) => Num::U(*v),
            _ => Num::I(0),
        }
    }

    /// Order-preserving coarse u64 key for vectorized pruning. Within one
    /// value family, `a <= b` implies `key(a) <= key(b)`, so a strict key
    /// inequality proves a strict value inequality and pruning on strict
    /// key comparisons is safe. Equal keys prove nothing, the caller must
    /// treat them as MayMatch. Null has no key
    pub fn stats_key(&self) -> Option<u64> {
        match self {
            LakeValue::Null => None,
            LakeValue::Bool(b) => Some(*b as u64),
            LakeValue::Int(v) => Some((*v as u64) ^ (1u64 << 63)),
            LakeValue::Int128(v) => Some((((*v as u128) ^ (1u128 << 127)) >> 64) as u64),
            LakeValue::UInt(v) => Some(*v),
            LakeValue::UInt128(v) => Some((*v >> 64) as u64),
            LakeValue::Float(v) => {
                let bits = v.to_bits();
                Some(if bits & (1u64 << 63) != 0 {
                    !bits
                } else {
                    bits | (1u64 << 63)
                })
            }
            LakeValue::Str(s) => Some(prefix_key(s.as_bytes())),
            LakeValue::Bytes(b) => Some(prefix_key(b)),
        }
    }
}

enum Num {
    I(i128),
    U(u128),
}

/// Compares a signed integer against a numeric value, None when the value
/// is not numeric. Used by cell-level comparison without allocating
pub(crate) fn compare_i128_to_value(x: i128, v: &LakeValue) -> Option<Ordering> {
    match v {
        LakeValue::Int(_) | LakeValue::Int128(_) | LakeValue::UInt(_) | LakeValue::UInt128(_) => {
            Some(compare_integers(Num::I(x), v.as_num()))
        }
        _ => None,
    }
}

/// Compares an unsigned integer against a numeric value
pub(crate) fn compare_u128_to_value(x: u128, v: &LakeValue) -> Option<Ordering> {
    match v {
        LakeValue::Int(_) | LakeValue::Int128(_) | LakeValue::UInt(_) | LakeValue::UInt128(_) => {
            Some(compare_integers(Num::U(x), v.as_num()))
        }
        _ => None,
    }
}

fn compare_integers(a: Num, b: Num) -> Ordering {
    match (a, b) {
        (Num::I(x), Num::I(y)) => x.cmp(&y),
        (Num::U(x), Num::U(y)) => x.cmp(&y),
        (Num::I(x), Num::U(y)) => {
            if x < 0 {
                Ordering::Less
            } else {
                (x as u128).cmp(&y)
            }
        }
        (Num::U(x), Num::I(y)) => {
            if y < 0 {
                Ordering::Greater
            } else {
                x.cmp(&(y as u128))
            }
        }
    }
}

/// Big endian first 8 bytes, zero padded, preserving lexicographic order
fn prefix_key(bytes: &[u8]) -> u64 {
    let mut key = [0u8; 8];
    let n = bytes.len().min(8);
    key[..n].copy_from_slice(&bytes[..n]);
    u64::from_be_bytes(key)
}

/// Comparison operator with SQL three-valued semantics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl CompareOp {
    /// Exact logical negation, `NOT (a < b)` is `a >= b` under SQL nulls
    pub fn negated(self) -> Self {
        match self {
            CompareOp::Eq => CompareOp::NotEq,
            CompareOp::NotEq => CompareOp::Eq,
            CompareOp::Lt => CompareOp::GtEq,
            CompareOp::LtEq => CompareOp::Gt,
            CompareOp::Gt => CompareOp::LtEq,
            CompareOp::GtEq => CompareOp::Lt,
        }
    }

    /// Operand swap, `a < b` is `b > a`
    pub fn flipped(self) -> Self {
        match self {
            CompareOp::Lt => CompareOp::Gt,
            CompareOp::Gt => CompareOp::Lt,
            CompareOp::LtEq => CompareOp::GtEq,
            CompareOp::GtEq => CompareOp::LtEq,
            other => other,
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            CompareOp::Eq => 0,
            CompareOp::NotEq => 1,
            CompareOp::Lt => 2,
            CompareOp::LtEq => 3,
            CompareOp::Gt => 4,
            CompareOp::GtEq => 5,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => CompareOp::Eq,
            1 => CompareOp::NotEq,
            2 => CompareOp::Lt,
            3 => CompareOp::LtEq,
            4 => CompareOp::Gt,
            5 => CompareOp::GtEq,
            _ => return None,
        })
    }
}

/// Predicate tree over lake columns addressed by column id
#[derive(Debug, Clone, PartialEq)]
pub enum LakePredicate {
    Compare {
        column_id: u32,
        op: CompareOp,
        value: LakeValue,
    },
    IsNull {
        column_id: u32,
    },
    IsNotNull {
        column_id: u32,
    },
    /// Membership, equivalent to an OR of equalities
    In {
        column_id: u32,
        values: Vec<LakeValue>,
    },
    And(Vec<LakePredicate>),
    Or(Vec<LakePredicate>),
    Not(Box<LakePredicate>),
}

/// Outcome of evaluating a predicate against one file's statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneDecision {
    /// No row in the file can satisfy the predicate, skip the file
    CannotMatch,
    /// The statistics cannot decide, scan the file
    MayMatch,
    /// Every row in the file satisfies the predicate, a delete with this
    /// predicate drops the file whole
    FullyCovers,
}

/// Per-column bounds a stats provider exposes for one file
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnBounds {
    /// Smallest non-null value, None when unknown or the column is all null
    pub min: Option<LakeValue>,
    /// Largest non-null value, None when unknown or the column is all null
    pub max: Option<LakeValue>,
    pub null_count: u64,
    pub row_count: u64,
}

/// Source of per-column bounds for one file. Returns references so a
/// pruning sweep over many files allocates nothing
pub trait StatsSource {
    /// None when the file carries no statistics for this column
    fn bounds(&self, column_id: u32) -> Option<&ColumnBounds>;

    /// False only when the column's value bloom proves the constant absent.
    ///
    /// Defaults to true, which prunes nothing. A source answers false only
    /// from a bloom it can encode the constant for, because a false negative
    /// here drops rows that do exist.
    fn may_contain(&self, _column_id: u32, _value: &LakeValue) -> bool {
        true
    }
}

// Nesting bound for decode, deep enough for any planner output and shallow
// enough that a corrupt file cannot overflow the stack
const MAX_PREDICATE_DEPTH: usize = 64;

// Smallest possible encoded predicate, an And with zero children
const MIN_ENCODED_NODE: usize = 5;

impl LakePredicate {
    /// Evaluates against one file's statistics, erring toward MayMatch
    pub fn prune(&self, stats: &dyn StatsSource) -> PruneDecision {
        self.prune_inner(stats, false)
    }

    /// Evaluates the predicate, or its exact negation when `negated`.
    ///
    /// Negation is carried as a flag rather than applied to a rewritten
    /// tree, so a Not costs nothing to prune however deep it sits and a
    /// sweep across a hundred thousand files allocates nothing at all.
    /// Each arm below is the same truth table `negated()` produces
    fn prune_inner(&self, stats: &dyn StatsSource, negated: bool) -> PruneDecision {
        match self {
            LakePredicate::Compare {
                column_id,
                op,
                value,
            } => {
                let op = if negated { op.negated() } else { *op };
                prune_leaf(stats, *column_id, op, value)
            }
            LakePredicate::IsNull { column_id } => {
                if negated {
                    prune_is_not_null(stats, *column_id)
                } else {
                    prune_is_null(stats, *column_id)
                }
            }
            LakePredicate::IsNotNull { column_id } => {
                if negated {
                    prune_is_null(stats, *column_id)
                } else {
                    prune_is_not_null(stats, *column_id)
                }
            }
            // NOT IN is the conjunction of the inequalities, IN the
            // disjunction of the equalities
            LakePredicate::In { column_id, values } if negated => {
                let mut all_cover = true;
                for v in values {
                    match prune_leaf(stats, *column_id, CompareOp::NotEq, v) {
                        PruneDecision::CannotMatch => return PruneDecision::CannotMatch,
                        PruneDecision::MayMatch => all_cover = false,
                        PruneDecision::FullyCovers => {}
                    }
                }
                if all_cover {
                    PruneDecision::FullyCovers
                } else {
                    PruneDecision::MayMatch
                }
            }
            LakePredicate::In { column_id, values } => {
                if values.is_empty() {
                    return PruneDecision::CannotMatch;
                }
                match stats.bounds(*column_id) {
                    Some(b) => {
                        let mut any_may = false;
                        for v in values {
                            match prune_compare(CompareOp::Eq, v, b) {
                                PruneDecision::FullyCovers => return PruneDecision::FullyCovers,
                                PruneDecision::MayMatch => {
                                    if stats.may_contain(*column_id, v) {
                                        any_may = true;
                                    }
                                }
                                PruneDecision::CannotMatch => {}
                            }
                        }
                        if any_may {
                            PruneDecision::MayMatch
                        } else {
                            PruneDecision::CannotMatch
                        }
                    }
                    None => PruneDecision::MayMatch,
                }
            }
            LakePredicate::And(children) | LakePredicate::Or(children) => {
                let conjunction = matches!(self, LakePredicate::And(_)) != negated;
                if conjunction {
                    let mut all_cover = true;
                    for child in children {
                        match child.prune_inner(stats, negated) {
                            PruneDecision::CannotMatch => return PruneDecision::CannotMatch,
                            PruneDecision::MayMatch => all_cover = false,
                            PruneDecision::FullyCovers => {}
                        }
                    }
                    if all_cover {
                        PruneDecision::FullyCovers
                    } else {
                        PruneDecision::MayMatch
                    }
                } else {
                    let mut all_cannot = true;
                    for child in children {
                        match child.prune_inner(stats, negated) {
                            PruneDecision::FullyCovers => return PruneDecision::FullyCovers,
                            PruneDecision::MayMatch => all_cannot = false,
                            PruneDecision::CannotMatch => {}
                        }
                    }
                    if all_cannot {
                        PruneDecision::CannotMatch
                    } else {
                        PruneDecision::MayMatch
                    }
                }
            }
            LakePredicate::Not(inner) => inner.prune_inner(stats, !negated),
        }
    }

    /// Working slots one pruning sweep needs, one byte per file each.
    ///
    /// A Not costs no slot because negation is carried as a flag rather
    /// than as a rewritten node
    pub fn eval_slots(&self) -> usize {
        match self {
            LakePredicate::Compare { .. }
            | LakePredicate::IsNull { .. }
            | LakePredicate::IsNotNull { .. } => 1,
            // The membership sweep folds each value's mask into an accumulator
            LakePredicate::In { .. } => 2,
            LakePredicate::And(children) | LakePredicate::Or(children) => {
                1 + children.iter().map(|c| c.eval_slots()).max().unwrap_or(0)
            }
            LakePredicate::Not(inner) => inner.eval_slots(),
        }
    }

    /// Exact logical negation under SQL three-valued semantics. Comparisons
    /// flip their operator, IsNull and IsNotNull swap, And and Or apply
    /// De Morgan, In expands to a conjunction of inequalities, and double
    /// negation cancels. The result contains no Not node
    pub fn negated(self) -> LakePredicate {
        match self {
            LakePredicate::Compare {
                column_id,
                op,
                value,
            } => LakePredicate::Compare {
                column_id,
                op: op.negated(),
                value,
            },
            LakePredicate::IsNull { column_id } => LakePredicate::IsNotNull { column_id },
            LakePredicate::IsNotNull { column_id } => LakePredicate::IsNull { column_id },
            LakePredicate::In { column_id, values } => LakePredicate::And(
                values
                    .into_iter()
                    .map(|v| LakePredicate::Compare {
                        column_id,
                        op: CompareOp::NotEq,
                        value: v,
                    })
                    .collect(),
            ),
            LakePredicate::And(children) => {
                LakePredicate::Or(children.into_iter().map(|c| c.negated()).collect())
            }
            LakePredicate::Or(children) => {
                LakePredicate::And(children.into_iter().map(|c| c.negated()).collect())
            }
            LakePredicate::Not(inner) => *inner,
        }
    }

    /// Column ids the predicate references, deduplicated
    pub fn referenced_columns(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        self.collect_columns(&mut ids);
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn collect_columns(&self, ids: &mut Vec<u32>) {
        match self {
            LakePredicate::Compare { column_id, .. }
            | LakePredicate::IsNull { column_id }
            | LakePredicate::IsNotNull { column_id }
            | LakePredicate::In { column_id, .. } => ids.push(*column_id),
            LakePredicate::And(children) | LakePredicate::Or(children) => {
                for c in children {
                    c.collect_columns(ids);
                }
            }
            LakePredicate::Not(inner) => inner.collect_columns(ids),
        }
    }

    /// Serializes the predicate tree, little endian throughout
    pub fn encode_into(&self, buf: &mut Vec<u8>) {
        match self {
            LakePredicate::Compare {
                column_id,
                op,
                value,
            } => {
                buf.push(1);
                buf.extend_from_slice(&column_id.to_le_bytes());
                buf.push(op.to_u8());
                encode_value(value, buf);
            }
            LakePredicate::IsNull { column_id } => {
                buf.push(2);
                buf.extend_from_slice(&column_id.to_le_bytes());
            }
            LakePredicate::IsNotNull { column_id } => {
                buf.push(3);
                buf.extend_from_slice(&column_id.to_le_bytes());
            }
            LakePredicate::In { column_id, values } => {
                buf.push(4);
                buf.extend_from_slice(&column_id.to_le_bytes());
                buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
                for v in values {
                    encode_value(v, buf);
                }
            }
            LakePredicate::And(children) => {
                buf.push(5);
                buf.extend_from_slice(&(children.len() as u32).to_le_bytes());
                for c in children {
                    c.encode_into(buf);
                }
            }
            LakePredicate::Or(children) => {
                buf.push(6);
                buf.extend_from_slice(&(children.len() as u32).to_le_bytes());
                for c in children {
                    c.encode_into(buf);
                }
            }
            LakePredicate::Not(inner) => {
                buf.push(7);
                inner.encode_into(buf);
            }
        }
    }

    /// Parses one predicate from the front of `bytes`, returning it and the
    /// number of bytes consumed. `ctx` names the enclosing file for errors
    pub fn decode(bytes: &[u8], ctx: &str) -> Result<(Self, usize), ZyronError> {
        let mut r = Cursor::new(bytes, ctx);
        let p = decode_node(&mut r, 0)?;
        Ok((p, r.pos()))
    }

    /// Parses one predicate from a cursor already positioned at its tag,
    /// used by the manifest codec to embed predicates inside a section
    pub(crate) fn decode_from(r: &mut Cursor<'_>) -> Result<Self, ZyronError> {
        decode_node(r, 0)
    }

    /// Stable 64-bit FNV-1a hash of the encoded form, used by the commit
    /// header's read-predicate field so conflict checks compare one integer
    pub fn stable_hash(&self) -> u64 {
        let mut buf = Vec::with_capacity(64);
        self.encode_into(&mut buf);
        let mut h = 0xcbf29ce484222325u64;
        for b in &buf {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

/// One comparison against one file's bounds, refined by the column's value
/// bloom.
///
/// The bloom answers equality only, and only downgrades a MayMatch. Bounds
/// proving full coverage outrank it, since a bloom disagreeing there is a
/// corrupt filter, not a smaller answer
fn prune_leaf(
    stats: &dyn StatsSource,
    column_id: u32,
    op: CompareOp,
    value: &LakeValue,
) -> PruneDecision {
    match stats.bounds(column_id) {
        Some(b) => {
            let decision = prune_compare(op, value, b);
            if op == CompareOp::Eq
                && decision == PruneDecision::MayMatch
                && !stats.may_contain(column_id, value)
            {
                PruneDecision::CannotMatch
            } else {
                decision
            }
        }
        None => PruneDecision::MayMatch,
    }
}

fn prune_is_null(stats: &dyn StatsSource, column_id: u32) -> PruneDecision {
    match stats.bounds(column_id) {
        Some(b) if b.null_count == 0 => PruneDecision::CannotMatch,
        Some(b) if b.null_count == b.row_count && b.row_count > 0 => PruneDecision::FullyCovers,
        _ => PruneDecision::MayMatch,
    }
}

fn prune_is_not_null(stats: &dyn StatsSource, column_id: u32) -> PruneDecision {
    match stats.bounds(column_id) {
        Some(b) if b.null_count == 0 && b.row_count > 0 => PruneDecision::FullyCovers,
        Some(b) if b.null_count == b.row_count && b.row_count > 0 => PruneDecision::CannotMatch,
        _ => PruneDecision::MayMatch,
    }
}

fn prune_compare(op: CompareOp, value: &LakeValue, b: &ColumnBounds) -> PruneDecision {
    // A comparison against null is unknown for every row
    if matches!(value, LakeValue::Null) {
        return PruneDecision::CannotMatch;
    }
    // No non-null values recorded. If the whole column is null nothing can
    // satisfy a comparison, otherwise the bounds are simply unknown
    let (min, max) = match (&b.min, &b.max) {
        (Some(min), Some(max)) => (min, max),
        _ => {
            return if b.row_count > 0 && b.null_count == b.row_count {
                PruneDecision::CannotMatch
            } else {
                PruneDecision::MayMatch
            };
        }
    };
    let (cmp_min, cmp_max) = match (value.compare(min), value.compare(max)) {
        (Some(a), Some(b)) => (a, b),
        _ => return PruneDecision::MayMatch,
    };
    let null_free = b.null_count == 0 && b.row_count > 0;
    match op {
        CompareOp::Eq => {
            if cmp_min == Ordering::Less || cmp_max == Ordering::Greater {
                PruneDecision::CannotMatch
            } else if cmp_min == Ordering::Equal && cmp_max == Ordering::Equal && null_free {
                PruneDecision::FullyCovers
            } else {
                PruneDecision::MayMatch
            }
        }
        CompareOp::NotEq => {
            if cmp_min == Ordering::Equal && cmp_max == Ordering::Equal {
                PruneDecision::CannotMatch
            } else if (cmp_min == Ordering::Less || cmp_max == Ordering::Greater) && null_free {
                PruneDecision::FullyCovers
            } else {
                PruneDecision::MayMatch
            }
        }
        // col < v, no match when min >= v, full cover when max < v
        CompareOp::Lt => {
            if cmp_min != Ordering::Greater {
                PruneDecision::CannotMatch
            } else if cmp_max == Ordering::Greater && null_free {
                PruneDecision::FullyCovers
            } else {
                PruneDecision::MayMatch
            }
        }
        // col <= v, no match when min > v, full cover when max <= v
        CompareOp::LtEq => {
            if cmp_min == Ordering::Less {
                PruneDecision::CannotMatch
            } else if cmp_max != Ordering::Less && null_free {
                PruneDecision::FullyCovers
            } else {
                PruneDecision::MayMatch
            }
        }
        // col > v, no match when max <= v, full cover when min > v
        CompareOp::Gt => {
            if cmp_max != Ordering::Less {
                PruneDecision::CannotMatch
            } else if cmp_min == Ordering::Less && null_free {
                PruneDecision::FullyCovers
            } else {
                PruneDecision::MayMatch
            }
        }
        // col >= v, no match when max < v, full cover when min >= v
        CompareOp::GtEq => {
            if cmp_max == Ordering::Greater {
                PruneDecision::CannotMatch
            } else if cmp_min != Ordering::Greater && null_free {
                PruneDecision::FullyCovers
            } else {
                PruneDecision::MayMatch
            }
        }
    }
}

pub(crate) fn encode_value(value: &LakeValue, buf: &mut Vec<u8>) {
    match value {
        LakeValue::Null => buf.push(0),
        LakeValue::Bool(v) => {
            buf.push(1);
            buf.push(*v as u8);
        }
        LakeValue::Int(v) => {
            buf.push(2);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        LakeValue::Int128(v) => {
            buf.push(3);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        LakeValue::UInt(v) => {
            buf.push(4);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        LakeValue::UInt128(v) => {
            buf.push(5);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        LakeValue::Float(v) => {
            buf.push(6);
            buf.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        LakeValue::Str(s) => {
            buf.push(7);
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        LakeValue::Bytes(b) => {
            buf.push(8);
            buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
            buf.extend_from_slice(b);
        }
    }
}

pub(crate) fn decode_value(r: &mut Cursor<'_>) -> Result<LakeValue, ZyronError> {
    let tag = r.u8()?;
    Ok(match tag {
        0 => LakeValue::Null,
        1 => match r.u8()? {
            0 => LakeValue::Bool(false),
            1 => LakeValue::Bool(true),
            v => return Err(r.corrupt(format!("invalid bool value byte {}", v))),
        },
        2 => LakeValue::Int(r.i64()?),
        3 => LakeValue::Int128(i128::from_le_bytes(r.array::<16>()?)),
        4 => LakeValue::UInt(r.u64()?),
        5 => LakeValue::UInt128(u128::from_le_bytes(r.array::<16>()?)),
        6 => LakeValue::Float(f64::from_bits(r.u64()?)),
        7 => {
            let len = r.u32()? as usize;
            LakeValue::Str(r.utf8(len, "predicate string")?)
        }
        8 => {
            let len = r.u32()? as usize;
            LakeValue::Bytes(r.take(len)?.to_vec())
        }
        v => return Err(r.corrupt(format!("unknown predicate value tag {}", v))),
    })
}

fn decode_node(r: &mut Cursor<'_>, depth: usize) -> Result<LakePredicate, ZyronError> {
    if depth > MAX_PREDICATE_DEPTH {
        return Err(r.corrupt(format!("predicate nesting exceeds {}", MAX_PREDICATE_DEPTH)));
    }
    let tag = r.u8()?;
    Ok(match tag {
        1 => {
            let column_id = r.u32()?;
            let op_raw = r.u8()?;
            let op = CompareOp::from_u8(op_raw)
                .ok_or_else(|| r.corrupt(format!("unknown compare op {}", op_raw)))?;
            let value = decode_value(r)?;
            LakePredicate::Compare {
                column_id,
                op,
                value,
            }
        }
        2 => LakePredicate::IsNull {
            column_id: r.u32()?,
        },
        3 => LakePredicate::IsNotNull {
            column_id: r.u32()?,
        },
        4 => {
            let column_id = r.u32()?;
            let count = r.u32()? as usize;
            r.check_count(count, 1, "IN value")?;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(decode_value(r)?);
            }
            LakePredicate::In { column_id, values }
        }
        5 | 6 => {
            let count = r.u32()? as usize;
            r.check_count(count, MIN_ENCODED_NODE, "predicate child")?;
            let mut children = Vec::with_capacity(count);
            for _ in 0..count {
                children.push(decode_node(r, depth + 1)?);
            }
            if tag == 5 {
                LakePredicate::And(children)
            } else {
                LakePredicate::Or(children)
            }
        }
        7 => LakePredicate::Not(Box::new(decode_node(r, depth + 1)?)),
        v => return Err(r.corrupt(format!("unknown predicate tag {}", v))),
    })
}

// ---------------------------------------------------------------------------
// Cluster fit
// ---------------------------------------------------------------------------

/// How much a table's layout does for one predicate.
///
/// Clustering sorts files by its keys in order, so a file's bounds on the
/// leading key are the narrowest the layout produces and every key after it
/// only narrows within a run of files that share a leading value. A
/// predicate that names none of the keys gets whatever bounds its column
/// happens to have, which is the layout doing nothing for it.
///
/// This is a statement about the layout, not a prediction of how many files
/// a scan will read. What actually got pruned is counted by the scan and
/// reported beside this
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterFit {
    /// Constrains the leading cluster key
    Good,
    /// Constrains a cluster key that is not the leading one
    Fair,
    /// Constrains no cluster key
    Poor,
}

impl ClusterFit {
    pub fn as_str(self) -> &'static str {
        match self {
            ClusterFit::Good => "good",
            ClusterFit::Fair => "fair",
            ClusterFit::Poor => "poor",
        }
    }
}

impl std::fmt::Display for ClusterFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a predicate gets from a table's layout, and which column decided it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterFitEstimate {
    pub fit: ClusterFit,
    /// The cluster key the predicate reached, or the column it constrained
    /// instead when it reached none. None only when the predicate
    /// constrains nothing at all
    pub column_id: Option<u32>,
    /// Where that key sits in the layout, zero for the leading one. None
    /// when the predicate reached no key
    pub position: Option<usize>,
}

/// Estimates what a predicate gets from a set of cluster keys.
///
/// `key_columns` is the layout in order, which is the accepted spec rather
/// than the declared one: a declared key measurement has replaced orders no
/// file, so judging a plan against it would report a fit the data does not
/// have.
///
/// Takes column ids rather than whole keys because the strategy decides how
/// values are bucketed within a key, not whether a predicate reaches it.
///
/// None when the table has no keys. A table nobody has laid out has no fit
/// to report, and calling that Poor would read as a defect rather than as
/// an absence
pub fn cluster_fit_estimate(
    key_columns: &[u32],
    predicate: &LakePredicate,
) -> Option<ClusterFitEstimate> {
    if key_columns.is_empty() {
        return None;
    }
    let referenced = predicate.referenced_columns();
    match key_columns.iter().position(|k| referenced.contains(k)) {
        Some(0) => Some(ClusterFitEstimate {
            fit: ClusterFit::Good,
            column_id: Some(key_columns[0]),
            position: Some(0),
        }),
        Some(position) => Some(ClusterFitEstimate {
            fit: ClusterFit::Fair,
            column_id: Some(key_columns[position]),
            position: Some(position),
        }),
        None => Some(ClusterFitEstimate {
            fit: ClusterFit::Poor,
            // What it constrained instead, which is the column an operator
            // would have to cluster by to turn this into a Good fit
            column_id: referenced.first().copied(),
            position: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MapStats(HashMap<u32, ColumnBounds>);

    impl StatsSource for MapStats {
        fn bounds(&self, column_id: u32) -> Option<&ColumnBounds> {
            self.0.get(&column_id)
        }
    }

    fn int_bounds(min: i64, max: i64, null_count: u64, row_count: u64) -> ColumnBounds {
        ColumnBounds {
            min: Some(LakeValue::Int(min)),
            max: Some(LakeValue::Int(max)),
            null_count,
            row_count,
        }
    }

    fn cmp(column_id: u32, op: CompareOp, v: i64) -> LakePredicate {
        LakePredicate::Compare {
            column_id,
            op,
            value: LakeValue::Int(v),
        }
    }

    #[test]
    fn test_range_pruning_including_negative_values() {
        // Pre-1970 timestamps are negative i64 microseconds, the sign flip
        // in the comparison and key paths must order them below zero
        let stats = MapStats(HashMap::from([(0, int_bounds(-5000, -100, 0, 10))]));

        assert_eq!(
            cmp(0, CompareOp::Gt, 0).prune(&stats),
            PruneDecision::CannotMatch
        );
        assert_eq!(
            cmp(0, CompareOp::Lt, 0).prune(&stats),
            PruneDecision::FullyCovers
        );
        assert_eq!(
            cmp(0, CompareOp::Lt, -200).prune(&stats),
            PruneDecision::MayMatch
        );
        assert_eq!(
            cmp(0, CompareOp::GtEq, -5000).prune(&stats),
            PruneDecision::FullyCovers
        );
        assert_eq!(
            cmp(0, CompareOp::Lt, -5000).prune(&stats),
            PruneDecision::CannotMatch
        );
        assert_eq!(
            cmp(0, CompareOp::LtEq, -5000).prune(&stats),
            PruneDecision::MayMatch
        );
    }

    #[test]
    fn test_equality_and_membership_pruning() {
        let stats = MapStats(HashMap::from([(0, int_bounds(10, 20, 0, 100))]));

        assert_eq!(
            cmp(0, CompareOp::Eq, 5).prune(&stats),
            PruneDecision::CannotMatch
        );
        assert_eq!(
            cmp(0, CompareOp::Eq, 15).prune(&stats),
            PruneDecision::MayMatch
        );

        let single = MapStats(HashMap::from([(0, int_bounds(7, 7, 0, 3))]));
        assert_eq!(
            cmp(0, CompareOp::Eq, 7).prune(&single),
            PruneDecision::FullyCovers
        );
        assert_eq!(
            cmp(0, CompareOp::NotEq, 7).prune(&single),
            PruneDecision::CannotMatch
        );
        assert_eq!(
            cmp(0, CompareOp::NotEq, 9).prune(&single),
            PruneDecision::FullyCovers
        );

        let none_in = LakePredicate::In {
            column_id: 0,
            values: vec![LakeValue::Int(1), LakeValue::Int(2)],
        };
        assert_eq!(none_in.prune(&stats), PruneDecision::CannotMatch);
        let some_in = LakePredicate::In {
            column_id: 0,
            values: vec![LakeValue::Int(1), LakeValue::Int(15)],
        };
        assert_eq!(some_in.prune(&stats), PruneDecision::MayMatch);
        let empty_in = LakePredicate::In {
            column_id: 0,
            values: vec![],
        };
        assert_eq!(empty_in.prune(&stats), PruneDecision::CannotMatch);
    }

    #[test]
    fn test_nulls_block_full_coverage_but_not_exclusion() {
        // 3 of 10 rows null. Exclusion still holds, coverage cannot
        let stats = MapStats(HashMap::from([(0, int_bounds(10, 20, 3, 10))]));
        assert_eq!(
            cmp(0, CompareOp::Lt, 5).prune(&stats),
            PruneDecision::CannotMatch
        );
        assert_eq!(
            cmp(0, CompareOp::Lt, 100).prune(&stats),
            PruneDecision::MayMatch
        );

        let all_null = MapStats(HashMap::from([(
            0,
            ColumnBounds {
                min: None,
                max: None,
                null_count: 10,
                row_count: 10,
            },
        )]));
        assert_eq!(
            cmp(0, CompareOp::Eq, 1).prune(&all_null),
            PruneDecision::CannotMatch
        );
        assert_eq!(
            LakePredicate::IsNull { column_id: 0 }.prune(&all_null),
            PruneDecision::FullyCovers
        );
        assert_eq!(
            LakePredicate::IsNotNull { column_id: 0 }.prune(&all_null),
            PruneDecision::CannotMatch
        );

        // Missing statistics never decide anything
        let empty = MapStats(HashMap::new());
        assert_eq!(
            cmp(9, CompareOp::Eq, 1).prune(&empty),
            PruneDecision::MayMatch
        );
        assert_eq!(
            LakePredicate::IsNull { column_id: 9 }.prune(&empty),
            PruneDecision::MayMatch
        );
    }

    #[test]
    fn test_and_or_not_combination() {
        let stats = MapStats(HashMap::from([
            (0, int_bounds(10, 20, 0, 100)),
            (1, int_bounds(-50, -30, 0, 100)),
        ]));

        let and = LakePredicate::And(vec![cmp(0, CompareOp::GtEq, 10), cmp(1, CompareOp::Lt, 0)]);
        assert_eq!(and.prune(&stats), PruneDecision::FullyCovers);

        let and_dead =
            LakePredicate::And(vec![cmp(0, CompareOp::GtEq, 10), cmp(1, CompareOp::Gt, 0)]);
        assert_eq!(and_dead.prune(&stats), PruneDecision::CannotMatch);

        let or = LakePredicate::Or(vec![cmp(0, CompareOp::Lt, 0), cmp(1, CompareOp::Lt, 0)]);
        assert_eq!(or.prune(&stats), PruneDecision::FullyCovers);

        let or_dead = LakePredicate::Or(vec![cmp(0, CompareOp::Lt, 0), cmp(1, CompareOp::Gt, 0)]);
        assert_eq!(or_dead.prune(&stats), PruneDecision::CannotMatch);

        // NOT of a full cover is an exclusion and the reverse
        let not_cover = LakePredicate::Not(Box::new(cmp(0, CompareOp::GtEq, 10)));
        assert_eq!(not_cover.prune(&stats), PruneDecision::CannotMatch);
        let not_dead = LakePredicate::Not(Box::new(cmp(0, CompareOp::Lt, 0)));
        assert_eq!(not_dead.prune(&stats), PruneDecision::FullyCovers);
    }

    #[test]
    fn test_negation_is_exact_and_removes_not_nodes() {
        let a = LakePredicate::And(vec![
            cmp(0, CompareOp::Lt, 5),
            LakePredicate::IsNull { column_id: 1 },
            LakePredicate::In {
                column_id: 2,
                values: vec![LakeValue::Int(1), LakeValue::Int(2)],
            },
        ]);
        // De Morgan with exact operator flips, In expands to a conjunction
        let expected = LakePredicate::Or(vec![
            cmp(0, CompareOp::GtEq, 5),
            LakePredicate::IsNotNull { column_id: 1 },
            LakePredicate::And(vec![
                cmp(2, CompareOp::NotEq, 1),
                cmp(2, CompareOp::NotEq, 2),
            ]),
        ]);
        assert_eq!(a.clone().negated(), expected);
        // A Not wrapper cancels against negation without touching the inner tree
        assert_eq!(LakePredicate::Not(Box::new(a.clone())).negated(), a);
    }

    #[test]
    fn test_integer_cross_signedness_comparison() {
        assert_eq!(
            LakeValue::Int(-1).compare(&LakeValue::UInt(0)),
            Some(Ordering::Less)
        );
        assert_eq!(
            LakeValue::UInt(u64::MAX).compare(&LakeValue::Int(5)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            LakeValue::UInt128(u128::MAX).compare(&LakeValue::Int128(i128::MAX)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            LakeValue::Int(7).compare(&LakeValue::Int128(7)),
            Some(Ordering::Equal)
        );
        // Cross-family outside the integers is not comparable
        assert_eq!(LakeValue::Int(1).compare(&LakeValue::Float(1.0)), None);
        assert_eq!(LakeValue::Null.compare(&LakeValue::Int(1)), None);
    }

    #[test]
    fn test_stats_key_is_monotone_per_family() {
        let ints = [i64::MIN, -5_000_000, -1, 0, 1, 42, i64::MAX];
        for w in ints.windows(2) {
            let a = LakeValue::Int(w[0]).stats_key().expect("key");
            let b = LakeValue::Int(w[1]).stats_key().expect("key");
            assert!(a < b, "int keys must order {} before {}", w[0], w[1]);
        }
        let floats = [f64::NEG_INFINITY, -1.5, -0.0, 0.0, 2.25, f64::INFINITY];
        for w in floats.windows(2) {
            let a = LakeValue::Float(w[0]).stats_key().expect("key");
            let b = LakeValue::Float(w[1]).stats_key().expect("key");
            assert!(a <= b, "float keys must order {} before {}", w[0], w[1]);
        }
        let strs = ["", "a", "ab", "b", "zzzzzzzzzz"];
        for w in strs.windows(2) {
            let a = LakeValue::Str(w[0].into()).stats_key().expect("key");
            let b = LakeValue::Str(w[1].into()).stats_key().expect("key");
            assert!(
                a <= b,
                "string keys must order {:?} before {:?}",
                w[0],
                w[1]
            );
        }
        let big = [0u128, 1 << 64, u128::MAX];
        for w in big.windows(2) {
            let a = LakeValue::UInt128(w[0]).stats_key().expect("key");
            let b = LakeValue::UInt128(w[1]).stats_key().expect("key");
            assert!(a <= b);
        }
        assert_eq!(LakeValue::Null.stats_key(), None);
    }

    #[test]
    fn test_codec_roundtrip_and_stable_hash() {
        let p = LakePredicate::Or(vec![
            LakePredicate::And(vec![
                cmp(0, CompareOp::GtEq, -12345),
                LakePredicate::Compare {
                    column_id: 1,
                    op: CompareOp::Eq,
                    value: LakeValue::Str("hello".into()),
                },
            ]),
            LakePredicate::Not(Box::new(LakePredicate::In {
                column_id: 2,
                values: vec![
                    LakeValue::Bool(true),
                    LakeValue::Float(-2.5),
                    LakeValue::Int128(i128::MIN),
                    LakeValue::UInt128(u128::MAX),
                    LakeValue::Bytes(vec![0, 255, 7]),
                    LakeValue::Null,
                ],
            })),
            LakePredicate::IsNull { column_id: 3 },
            LakePredicate::IsNotNull { column_id: 4 },
        ]);
        let mut buf = Vec::new();
        p.encode_into(&mut buf);
        let (decoded, consumed) = LakePredicate::decode(&buf, "test").expect("decodes");
        assert_eq!(decoded, p);
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded.stable_hash(), p.stable_hash());
        assert_ne!(p.stable_hash(), cmp(0, CompareOp::Eq, 1).stable_hash());
    }

    #[test]
    fn test_decode_rejects_corruption_and_deep_nesting() {
        let p = cmp(0, CompareOp::Eq, 5);
        let mut buf = Vec::new();
        p.encode_into(&mut buf);
        for cut in 0..buf.len() {
            assert!(LakePredicate::decode(&buf[..cut], "test").is_err());
        }
        let mut bad_tag = buf.clone();
        bad_tag[0] = 99;
        assert!(LakePredicate::decode(&bad_tag, "test").is_err());
        let mut bad_op = buf.clone();
        bad_op[5] = 99;
        assert!(LakePredicate::decode(&bad_op, "test").is_err());

        // A tower of Not tags one past the depth cap
        let mut deep = vec![7u8; MAX_PREDICATE_DEPTH + 1];
        deep.extend_from_slice(&buf);
        assert!(LakePredicate::decode(&deep, "test").is_err());

        // Child count far beyond what the buffer could hold
        let mut fake_and = vec![5u8];
        fake_and.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(LakePredicate::decode(&fake_and, "test").is_err());
    }

    #[test]
    fn test_referenced_columns_and_op_helpers() {
        let p = LakePredicate::And(vec![
            cmp(3, CompareOp::Eq, 1),
            LakePredicate::Or(vec![cmp(1, CompareOp::Lt, 2), cmp(3, CompareOp::Gt, 0)]),
            LakePredicate::Not(Box::new(LakePredicate::IsNull { column_id: 7 })),
        ]);
        assert_eq!(p.referenced_columns(), vec![1, 3, 7]);
        assert_eq!(CompareOp::Lt.negated(), CompareOp::GtEq);
        assert_eq!(CompareOp::Lt.flipped(), CompareOp::Gt);
        assert_eq!(CompareOp::Eq.flipped(), CompareOp::Eq);
        assert_eq!(CompareOp::NotEq.negated(), CompareOp::Eq);
    }

    /// Files are sorted by the leading key first, so a predicate that names
    /// it gets the narrowest bounds the layout can produce. One that names
    /// a later key gets bounds that only narrow inside a run of files
    /// sharing a leading value, and one that names no key gets nothing from
    /// the layout at all
    #[test]
    fn test_cluster_fit_reads_the_position_of_the_key_a_predicate_reaches() {
        let keys = [7u32, 3, 9];

        let leading = LakePredicate::Compare {
            column_id: 7,
            op: CompareOp::Eq,
            value: LakeValue::Int(1),
        };
        let fit = cluster_fit_estimate(&keys, &leading).expect("a laid out table has a fit");
        assert_eq!(fit.fit, ClusterFit::Good);
        assert_eq!(fit.column_id, Some(7));
        assert_eq!(fit.position, Some(0));

        let secondary = LakePredicate::In {
            column_id: 9,
            values: vec![LakeValue::Int(1), LakeValue::Int(2)],
        };
        let fit = cluster_fit_estimate(&keys, &secondary).expect("fit");
        assert_eq!(fit.fit, ClusterFit::Fair);
        assert_eq!(fit.column_id, Some(9));
        assert_eq!(fit.position, Some(2));

        let unrelated = LakePredicate::IsNull { column_id: 42 };
        let fit = cluster_fit_estimate(&keys, &unrelated).expect("fit");
        assert_eq!(fit.fit, ClusterFit::Poor);
        assert_eq!(
            fit.column_id,
            Some(42),
            "a poor fit has to name what the predicate constrained instead, which is the \
             column an operator would cluster by to fix it"
        );
        assert_eq!(fit.position, None);
    }

    /// A conjunction reaching the leading key is a good fit even when its
    /// other branches reach nothing, because the leading key is what
    /// decides which files are opened
    #[test]
    fn test_cluster_fit_takes_the_best_key_a_conjunction_reaches() {
        let keys = [7u32, 3];
        let predicate = LakePredicate::And(vec![
            LakePredicate::Compare {
                column_id: 42,
                op: CompareOp::Gt,
                value: LakeValue::Int(0),
            },
            LakePredicate::Compare {
                column_id: 3,
                op: CompareOp::Eq,
                value: LakeValue::Int(5),
            },
            LakePredicate::Compare {
                column_id: 7,
                op: CompareOp::Eq,
                value: LakeValue::Int(5),
            },
        ]);
        let fit = cluster_fit_estimate(&keys, &predicate).expect("fit");
        assert_eq!(fit.fit, ClusterFit::Good);
        assert_eq!(fit.column_id, Some(7));
    }

    /// A table nobody has laid out has no fit to report. Calling that poor
    /// would read as a defect in the plan rather than as the absence of a
    /// layout to judge it against
    #[test]
    fn test_a_table_with_no_cluster_keys_has_no_fit() {
        let predicate = LakePredicate::Compare {
            column_id: 1,
            op: CompareOp::Eq,
            value: LakeValue::Int(1),
        };
        assert!(cluster_fit_estimate(&[], &predicate).is_none());
    }
}
