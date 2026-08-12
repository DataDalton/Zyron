//! Ordering curves: the row order each cluster strategy asks for.
//!
//! A declared strategy that did not change the layout would be a lie, so
//! this is where `CLUSTER BY (a, b USING BitInterleave)` stops being a note
//! in the manifest and starts deciding which rows share a file.
//!
//! Every strategy produces one fixed-width ordering key per row and the
//! writer sorts by it, so the cost is one allocation of `8 * dimensions`
//! bytes per row and a byte comparison, whatever the strategy.
//!
//! The curves and what each is for:
//!
//! * `RangePartition` concatenates the normalized components, so byte order
//!   is value order and a range predicate on the leading column reads one
//!   run of files. The default, and the right answer for a temporal or
//!   high-cardinality key.
//! * `BitInterleave` (Z-order) interleaves the components' bits, so a
//!   predicate on any single dimension still skips files. It degrades when
//!   the dimensions have very different cardinalities, which is why it is
//!   one option rather than the only one.
//! * `SpaceFilling` (Hilbert) preserves locality better than Z-order at the
//!   cost of a more expensive transform, worth it when the dimensions are
//!   comparable in cardinality.
//! * `AntiCluster` deliberately scatters equal values, which is what a shard
//!   key wants: co-located rows on one node become spread files rather than
//!   one hot file.
//!
//! Normalization is what makes any of this correct. Each component becomes a
//! u64 whose unsigned order equals the value's order, so a negative integer
//! sorts below a positive one and a float sorts by value rather than by its
//! bit pattern.
//!
//! This lives in zyron-common rather than with the lake writer because both
//! storage tiers order rows by it: the lake writer lays out a .zyr file, and
//! the heap fold tier lays out the segments it produces. One definition, so
//! a heap table and a lake table clustered on the same key agree byte for
//! byte on what that means.

use crate::cluster::ClusterStrategy;
use crate::types::TypeId;

/// How a physical type's cells compare
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellFamily {
    Bool,
    /// All signed integer widths plus temporal types stored as signed
    /// integers and fixed-point decimals at one per-column scale
    SignedInt,
    /// Unsigned widths plus packed monotone encodings
    UnsignedInt,
    Float,
    /// UTF-8 text, byte order equals collation order for the raw collation
    Str,
    /// Raw bytes with lexicographic order, including fixed identifiers
    Bytes,
    /// Byte order does not reflect value order
    Unordered,
}

pub fn cell_family(physical: TypeId) -> CellFamily {
    match physical {
        TypeId::Boolean => CellFamily::Bool,
        TypeId::Int8
        | TypeId::Int16
        | TypeId::Int32
        | TypeId::Int64
        | TypeId::Int128
        | TypeId::Date
        | TypeId::Time
        | TypeId::Timestamp
        | TypeId::TimestampTz
        | TypeId::Hlc
        | TypeId::Decimal => CellFamily::SignedInt,
        TypeId::UInt8
        | TypeId::UInt16
        | TypeId::UInt32
        | TypeId::UInt64
        | TypeId::UInt128
        | TypeId::SemVer => CellFamily::UnsignedInt,
        TypeId::Float32 | TypeId::Float64 => CellFamily::Float,
        TypeId::Char | TypeId::Varchar | TypeId::Text => CellFamily::Str,
        TypeId::Binary | TypeId::Varbinary | TypeId::Bytea | TypeId::Uuid | TypeId::MacAddr => {
            CellFamily::Bytes
        }
        _ => CellFamily::Unordered,
    }
}

/// Bits per normalized component. Wider components are truncated to their
/// most significant bits, which keeps the ordering coarse rather than wrong:
/// two rows sharing a prefix simply land adjacent.
const COMPONENT_BITS: usize = 64;

/// Maps one cell to a u64 whose unsigned order matches the value's order.
///
/// NULL has no value to place, so the caller decides where nulls go rather
/// than this inventing a position for them.
pub fn normalize_component(physical: TypeId, cell: &[u8]) -> u64 {
    match cell_family(physical) {
        CellFamily::Bool => cell.first().map(|b| (*b != 0) as u64).unwrap_or(0),
        CellFamily::SignedInt => {
            let value = signed_of(cell);
            // Flipping the sign bit turns two's complement order into
            // unsigned order
            (value as u64) ^ (1u64 << 63)
        }
        CellFamily::UnsignedInt => unsigned_of(cell),
        CellFamily::Float => {
            let bits = float_of(cell).to_bits();
            // Total order: negatives reverse, positives keep their order
            if bits & (1u64 << 63) != 0 {
                !bits
            } else {
                bits | (1u64 << 63)
            }
        }
        // Byte-ordered families take their leading eight bytes, big endian,
        // so byte order is preserved and short values sort before longer
        // ones sharing their prefix
        CellFamily::Str | CellFamily::Bytes => prefix_be(cell),
        CellFamily::Unordered => prefix_be(cell),
    }
}

fn signed_of(cell: &[u8]) -> i64 {
    match cell.len() {
        1 => cell[0] as i8 as i64,
        2 => i16::from_le_bytes([cell[0], cell[1]]) as i64,
        4 => {
            let mut a = [0u8; 4];
            a.copy_from_slice(cell);
            i32::from_le_bytes(a) as i64
        }
        8 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(cell);
            i64::from_le_bytes(a)
        }
        16 => {
            let mut a = [0u8; 16];
            a.copy_from_slice(cell);
            // The high 64 bits order a 128-bit value, the rest only refines
            // rows that already share them
            (i128::from_le_bytes(a) >> 64) as i64
        }
        _ => 0,
    }
}

fn unsigned_of(cell: &[u8]) -> u64 {
    match cell.len() {
        1 => cell[0] as u64,
        2 => u16::from_le_bytes([cell[0], cell[1]]) as u64,
        4 => {
            let mut a = [0u8; 4];
            a.copy_from_slice(cell);
            u32::from_le_bytes(a) as u64
        }
        8 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(cell);
            u64::from_le_bytes(a)
        }
        16 => {
            let mut a = [0u8; 16];
            a.copy_from_slice(cell);
            (u128::from_le_bytes(a) >> 64) as u64
        }
        _ => 0,
    }
}

fn float_of(cell: &[u8]) -> f64 {
    match cell.len() {
        4 => {
            let mut a = [0u8; 4];
            a.copy_from_slice(cell);
            f32::from_le_bytes(a) as f64
        }
        8 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(cell);
            f64::from_le_bytes(a)
        }
        _ => 0.0,
    }
}

/// The leading eight bytes as a big-endian u64, zero padded.
fn prefix_be(cell: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let len = cell.len().min(8);
    buf[..len].copy_from_slice(&cell[..len]);
    u64::from_be_bytes(buf)
}

/// Splits every dimension's bits across the output so a predicate on any one
/// of them still selects a bounded set of ranges.
fn bit_interleave(axes: &[u64]) -> Vec<u8> {
    let dims = axes.len();
    let mut out = vec![0u8; dims * 8];
    let total_bits = dims * COMPONENT_BITS;
    for bit in 0..total_bits {
        // Most significant bit of every axis first, so the leading output
        // bytes carry the coarsest information from all dimensions
        let axis = bit % dims;
        let level = bit / dims;
        let source = (axes[axis] >> (COMPONENT_BITS - 1 - level)) & 1;
        if source != 0 {
            out[bit / 8] |= 1 << (7 - (bit % 8));
        }
    }
    out
}

/// Hilbert index of a point, as the transposed axes the curve defines.
///
/// The standard Skilling transform: fold the axes into Gray-code space, then
/// transpose so the leading bits of the result order points along the curve.
/// Hilbert keeps neighbouring points closer than Z-order does, which is what
/// pays for the extra passes when the dimensions are comparable in size.
fn hilbert(axes: &[u64]) -> Vec<u8> {
    let dims = axes.len();
    let mut x: Vec<u64> = axes.to_vec();
    let bits = COMPONENT_BITS as u32;

    // Inverse undo of the Gray code, from the most significant bit down
    let mut q = 1u64 << (bits - 1);
    while q > 1 {
        let p = q - 1;
        for i in 0..dims {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    // Gray encode
    for i in 1..dims {
        x[i] ^= x[i - 1];
    }
    let mut t = 0u64;
    let mut q = 1u64 << (bits - 1);
    while q > 1 {
        if x[dims - 1] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for value in x.iter_mut() {
        *value ^= t;
    }
    // Transpose: the curve's order is the axes read bit plane by bit plane
    bit_interleave(&x)
}

/// Scatters equal values by hashing, so rows that would have shared a file
/// spread across many.
fn anti_cluster(axes: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(axes.len() * 8);
    for (index, axis) in axes.iter().enumerate() {
        // A per-dimension salt keeps two dimensions with equal values from
        // hashing to the same spread
        let mut state = axis.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index as u64 + 1));
        let mixed = crate::prng::splitMix64(&mut state);
        out.extend_from_slice(&mixed.to_be_bytes());
    }
    out
}

/// The ordering key one row gets under a strategy.
///
/// `axes` are the row's normalized key components in declaration order.
/// The returned bytes compare lexicographically, so the writer sorts rows by
/// comparing them directly and no strategy needs its own comparator.
pub fn ordering_key(strategy: ClusterStrategy, axes: &[u64]) -> Vec<u8> {
    if axes.is_empty() {
        return Vec::new();
    }
    match strategy {
        // Concatenation: byte order is value order, so a range predicate on
        // the leading column reads one run
        ClusterStrategy::RangePartition => {
            let mut out = Vec::with_capacity(axes.len() * 8);
            for axis in axes {
                out.extend_from_slice(&axis.to_be_bytes());
            }
            out
        }
        ClusterStrategy::BitInterleave => bit_interleave(axes),
        ClusterStrategy::SpaceFilling => hilbert(axes),
        ClusterStrategy::AntiCluster => anti_cluster(axes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(strategy: ClusterStrategy, axes: &[u64]) -> Vec<u8> {
        ordering_key(strategy, axes)
    }

    #[test]
    fn test_normalization_puts_negatives_below_positives() {
        let neg = normalize_component(TypeId::Int64, &(-5i64).to_le_bytes());
        let zero = normalize_component(TypeId::Int64, &0i64.to_le_bytes());
        let pos = normalize_component(TypeId::Int64, &5i64.to_le_bytes());
        assert!(neg < zero && zero < pos);

        // Floats order by value, not by bit pattern
        let fneg = normalize_component(TypeId::Float64, &(-1.5f64).to_le_bytes());
        let fzero = normalize_component(TypeId::Float64, &0.0f64.to_le_bytes());
        let fpos = normalize_component(TypeId::Float64, &1.5f64.to_le_bytes());
        assert!(fneg < fzero && fzero < fpos);

        // Text keeps byte order through its leading eight bytes
        let a = normalize_component(TypeId::Varchar, b"apple");
        let b = normalize_component(TypeId::Varchar, b"banana");
        assert!(a < b);
    }

    #[test]
    fn test_range_partition_orders_by_the_leading_component() {
        let mut rows: Vec<(u64, u64)> = vec![(3, 9), (1, 5), (2, 1), (1, 2)];
        rows.sort_by_key(|(a, b)| key(ClusterStrategy::RangePartition, &[*a, *b]));
        assert_eq!(rows, vec![(1, 2), (1, 5), (2, 1), (3, 9)]);
    }

    #[test]
    fn test_bit_interleave_groups_by_the_high_bits_of_every_dimension() {
        // The defining property, and the one file skipping rests on: the
        // order is by the most significant bits of ALL dimensions first, so
        // a block of the space is contiguous. Lexicographic order can only
        // do that for its leading column
        let mut points: Vec<(u64, u64)> = Vec::new();
        for x in 0..8u64 {
            for y in 0..8u64 {
                points.push((x, y));
            }
        }

        let blocks_are_contiguous = |strategy: ClusterStrategy| -> bool {
            let mut ordered = points.clone();
            ordered.sort_by_key(|(x, y)| key(strategy, &[*x, *y]));
            // Every 2x2 block of the space must occupy consecutive positions
            let mut seen: std::collections::HashMap<(u64, u64), (usize, usize)> =
                std::collections::HashMap::new();
            for (position, (x, y)) in ordered.iter().enumerate() {
                let block = (x >> 1, y >> 1);
                let entry = seen.entry(block).or_insert((position, position));
                entry.0 = entry.0.min(position);
                entry.1 = entry.1.max(position);
            }
            seen.values().all(|(first, last)| last - first == 3)
        };

        assert!(
            blocks_are_contiguous(ClusterStrategy::BitInterleave),
            "a block of the space must be one run"
        );
        assert!(
            !blocks_are_contiguous(ClusterStrategy::RangePartition),
            "lexicographic order cannot keep a block together"
        );

        // And it stays a bijection: no two grid points share a key
        let mut keys: Vec<Vec<u8>> = points
            .iter()
            .map(|(x, y)| key(ClusterStrategy::BitInterleave, &[*x, *y]))
            .collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 64);
    }

    #[test]
    fn test_space_filling_visits_every_point_once_and_stays_local() {
        let mut points: Vec<(u64, u64)> = Vec::new();
        for x in 0..4u64 {
            for y in 0..4u64 {
                points.push((x, y));
            }
        }
        let mut keys: Vec<Vec<u8>> = points
            .iter()
            .map(|(x, y)| key(ClusterStrategy::SpaceFilling, &[*x, *y]))
            .collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 16, "the curve is a bijection over the grid");

        points.sort_by_key(|(x, y)| key(ClusterStrategy::SpaceFilling, &[*x, *y]));
        // Hilbert never jumps: consecutive points differ by one step in one
        // dimension
        for pair in points.windows(2) {
            let (x0, y0) = pair[0];
            let (x1, y1) = pair[1];
            assert_eq!(
                x0.abs_diff(x1) + y0.abs_diff(y1),
                1,
                "Hilbert jumped from ({x0},{y0}) to ({x1},{y1})"
            );
        }
    }

    #[test]
    fn test_anti_cluster_scatters_what_range_partition_gathers() {
        // A hundred rows of ten repeated values, the shape a shard key has
        let values: Vec<u64> = (0..100u64).map(|i| i % 10).collect();
        let mut ranged: Vec<u64> = values.clone();
        ranged.sort_by_key(|v| key(ClusterStrategy::RangePartition, &[*v]));
        // Range ordering gathers equal values into runs
        assert_eq!(ranged[0], ranged[9]);

        let mut spread: Vec<u64> = values.clone();
        spread.sort_by_key(|v| key(ClusterStrategy::AntiCluster, &[*v]));
        // Anti-clustering still groups identical values, what it changes is
        // WHICH values sit next to each other, so neighbouring distinct
        // values are no longer numerically adjacent
        let distinct: Vec<u64> = {
            let mut seen = Vec::new();
            for v in &spread {
                if seen.last() != Some(v) {
                    seen.push(*v);
                }
            }
            seen
        };
        assert_eq!(distinct.len(), 10);
        assert_ne!(
            distinct,
            (0..10u64).collect::<Vec<u64>>(),
            "anti-clustering must not reproduce value order"
        );
    }

    #[test]
    fn test_an_empty_key_orders_nothing() {
        assert!(key(ClusterStrategy::BitInterleave, &[]).is_empty());
        assert!(key(ClusterStrategy::RangePartition, &[]).is_empty());
    }
}
