//! Column segment: the atomic unit of columnar storage.
//!
//! A ColumnSegment holds one column's data for a contiguous range of rows,
//! including encoding metadata, zone maps for segment pruning, and an
//! optional bloom filter for point lookups.

use crate::columnar::bloom::BloomFilter;
use crate::columnar::constants::*;
use crate::columnar::sketch::DistinctSketch;
use crate::encoding::{
    ColumnSampleStats, EncodingType, cardinality_cap, create_encoding, select_encoding_prepared,
    select_encoding_varlen, varlen_pack,
};
use zyron_common::types::TypeId;
use zyron_common::{Result, ZyronError};

/// On-disk segment header (128 bytes). Describes a single column's data
/// within a .zyr file, including encoding, statistics, and offsets to
/// the encoded payload and bloom filter.
#[derive(Debug, Clone)]
pub struct SegmentHeader {
    /// Column ordinal within the table schema.
    pub column_id: u32,
    /// Encoding strategy applied to this segment's data.
    pub encoding_type: EncodingType,
    /// Byte size of column data before encoding.
    pub raw_size: u64,
    /// Byte size of encoded column data.
    pub encoded_size: u64,
    /// Number of null values in this segment.
    pub null_count: u64,
    /// Number of distinct non-null values.
    pub cardinality: u64,
    /// Minimum value in the segment (left-padded with zeros).
    pub min_value: [u8; STAT_VALUE_SIZE],
    /// Maximum value in the segment (left-padded with zeros).
    pub max_value: [u8; STAT_VALUE_SIZE],
    /// CRC of this segment's encoded payload. The scan verifies it over the
    /// bytes it reads to decode anyway, so corruption is detected with zero
    /// extra IO. A metadata aggregate never reads the payload, so it never
    /// pays this.
    pub data_checksum: u32,
    /// CRC of the segment header itself (bytes [0..108]). Lets a metadata
    /// aggregate trust min/max/null_count/encoded_size from a header-only
    /// read without a whole-file checksum pass.
    pub header_crc: u32,
    /// Byte offset from the start of this segment to its bloom filter, 0
    /// when the segment carries none. The segment builder cannot know where
    /// the writer will place it in the file, so the offset is segment
    /// relative and a reader adds the segment's file offset.
    pub bloom_filter_offset: u64,
    /// Size of bloom filter in bytes.
    pub bloom_filter_size: u32,
    /// Whether the segment's rows are sorted by value.
    pub is_sorted: bool,
}

impl SegmentHeader {
    /// Serializes this header into a 128-byte little-endian buffer.
    pub fn to_bytes(&self) -> [u8; SEGMENT_HEADER_SIZE] {
        let mut buf = [0u8; SEGMENT_HEADER_SIZE];

        buf[0..4].copy_from_slice(&self.column_id.to_le_bytes());
        buf[4] = self.encoding_type as u8;
        // [5..8] reserved
        buf[8..16].copy_from_slice(&self.raw_size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.encoded_size.to_le_bytes());
        buf[24..32].copy_from_slice(&self.null_count.to_le_bytes());
        buf[32..40].copy_from_slice(&self.cardinality.to_le_bytes());
        buf[40..72].copy_from_slice(&self.min_value);
        buf[72..104].copy_from_slice(&self.max_value);
        buf[104..108].copy_from_slice(&self.data_checksum.to_le_bytes());
        // [108..112] = header_crc, filled last over [0..108].
        buf[112..120].copy_from_slice(&self.bloom_filter_offset.to_le_bytes());
        buf[120..124].copy_from_slice(&self.bloom_filter_size.to_le_bytes());
        buf[124] = if self.is_sorted { 1 } else { 0 };
        // [125..128] reserved

        let hc = zyron_common::hash32(&buf[0..108]);
        buf[108..112].copy_from_slice(&hc.to_le_bytes());

        buf
    }

    /// Deserializes a 128-byte little-endian buffer into a SegmentHeader.
    pub fn from_bytes(buf: &[u8; SEGMENT_HEADER_SIZE]) -> Result<Self> {
        let columnId = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let encodingType = EncodingType::from_u8(buf[4])?;
        let rawSize = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);
        let encodedSize = u64::from_le_bytes([
            buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
        ]);
        let nullCount = u64::from_le_bytes([
            buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
        ]);
        let cardinality = u64::from_le_bytes([
            buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
        ]);

        let mut minValue = [0u8; STAT_VALUE_SIZE];
        minValue.copy_from_slice(&buf[40..72]);
        let mut maxValue = [0u8; STAT_VALUE_SIZE];
        maxValue.copy_from_slice(&buf[72..104]);

        let dataChecksum = u32::from_le_bytes([buf[104], buf[105], buf[106], buf[107]]);
        let headerCrc = u32::from_le_bytes([buf[108], buf[109], buf[110], buf[111]]);
        // Self-verify the header so a metadata aggregate can trust it from a
        // header-only read (no whole-file checksum pass).
        let computed = zyron_common::hash32(&buf[0..108]);
        if headerCrc != computed {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment header checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
                headerCrc, computed
            )));
        }
        let bloomFilterOffset = u64::from_le_bytes([
            buf[112], buf[113], buf[114], buf[115], buf[116], buf[117], buf[118], buf[119],
        ]);
        let bloomFilterSize = u32::from_le_bytes([buf[120], buf[121], buf[122], buf[123]]);
        let isSorted = buf[124] != 0;

        Ok(Self {
            column_id: columnId,
            encoding_type: encodingType,
            raw_size: rawSize,
            encoded_size: encodedSize,
            null_count: nullCount,
            cardinality,
            min_value: minValue,
            max_value: maxValue,
            data_checksum: dataChecksum,
            header_crc: headerCrc,
            bloom_filter_offset: bloomFilterOffset,
            bloom_filter_size: bloomFilterSize,
            is_sorted: isSorted,
        })
    }
}

/// Zone map entry (64 bytes). Stores the min and max value for a batch
/// of ZONE_MAP_BATCH_SIZE rows, enabling segment pruning during scans.
#[derive(Debug, Clone)]
pub struct ZoneMapEntry {
    /// Minimum value in this zone (left-padded with zeros).
    pub min_value: [u8; STAT_VALUE_SIZE],
    /// Maximum value in this zone (left-padded with zeros).
    pub max_value: [u8; STAT_VALUE_SIZE],
}

impl ZoneMapEntry {
    /// Serializes this zone map entry into a 64-byte buffer.
    pub fn to_bytes(&self) -> [u8; ZONE_MAP_ENTRY_SIZE] {
        let mut buf = [0u8; ZONE_MAP_ENTRY_SIZE];
        buf[0..STAT_VALUE_SIZE].copy_from_slice(&self.min_value);
        buf[STAT_VALUE_SIZE..ZONE_MAP_ENTRY_SIZE].copy_from_slice(&self.max_value);
        buf
    }

    /// Deserializes a 64-byte buffer into a ZoneMapEntry.
    pub fn from_bytes(buf: &[u8; ZONE_MAP_ENTRY_SIZE]) -> Self {
        let mut minValue = [0u8; STAT_VALUE_SIZE];
        minValue.copy_from_slice(&buf[0..STAT_VALUE_SIZE]);
        let mut maxValue = [0u8; STAT_VALUE_SIZE];
        maxValue.copy_from_slice(&buf[STAT_VALUE_SIZE..ZONE_MAP_ENTRY_SIZE]);
        Self {
            min_value: minValue,
            max_value: maxValue,
        }
    }
}

/// Whether a column segment carries a value bloom filter.
///
/// Auto is the cardinality heuristic: a bloom pays for itself only on a
/// column with enough distinct values, and a dictionary-encoded segment
/// already carries an exact lookup structure. Force overrides both, which is
/// what a declared `bloom_filter_columns` asks for, and Suppress refuses one
/// on a column that is never probed for equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BloomPolicy {
    #[default]
    Auto,
    Force,
    Suppress,
}

impl BloomPolicy {
    /// Decides whether this segment gets a bloom filter
    pub fn builds_bloom(self, cardinality: u64, encoding: EncodingType) -> bool {
        match self {
            Self::Suppress => false,
            // A forced bloom on a single-value column is still built, the
            // caller asked for a probe that answers without opening the file
            Self::Force => cardinality > 0,
            Self::Auto => {
                cardinality >= BLOOM_MIN_CARDINALITY && encoding != EncodingType::Dictionary
            }
        }
    }
}

/// Build-time choices a caller can make for one column segment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentOptions {
    /// Whether the segment carries a value bloom filter
    pub bloom: BloomPolicy,
    /// Pick the encoding by trial encoding every row instead of a bounded
    /// prefix. A caller holding the whole column already, like the lake
    /// writer, pays nothing extra for a decision that cannot be wrong on an
    /// unrepresentative prefix
    pub exact_encoding: bool,
    /// Count distinct values across the whole column with a fixed-size
    /// sketch and report the estimate on the segment. The bloom decision
    /// reads only the exact count capped at the bloom threshold, so a caller
    /// that publishes no distinct estimate leaves this off and the sketch is
    /// never built
    pub distinct_sketch: bool,
}

/// A fully materialized column segment ready for writing to a .zyr file.
/// Contains the segment header, encoded data, zone maps, null bitmap,
/// and an optional bloom filter (attached separately by the caller).
pub struct ColumnSegment {
    /// Segment metadata header.
    pub header: SegmentHeader,
    /// Bloom filter for point-lookup pruning. Built separately by the
    /// caller after segment construction, since not all segments need one.
    pub bloom_filter: Option<BloomFilter>,
    /// Zone maps for range-scan pruning, one per ZONE_MAP_BATCH_SIZE rows.
    pub zone_maps: Vec<ZoneMapEntry>,
    /// Encoded column data produced by the selected encoding strategy.
    pub encoded_data: Vec<u8>,
    /// Packed bit array marking null positions. Bit i is set if row i is null.
    /// Empty if no nulls exist.
    pub null_bitmap: Vec<u8>,
    /// Row holding the segment's smallest non-null value under the column's
    /// slot order, indexing the `values` slice the build was handed. None
    /// when every row is null. The header records the value as a 32-byte
    /// slot, so a caller needing the value itself, at its full width, reads
    /// it back through this row
    pub min_row: Option<usize>,
    /// Row holding the segment's largest non-null value, as `min_row`
    pub max_row: Option<usize>,
    /// Estimated distinct non-null values across the whole column, present
    /// when the caller asked for the sketch. The header's `cardinality` is
    /// capped at the bloom threshold and answers a different question
    pub ndv: Option<u64>,
}

/// Copies a value into a STAT_VALUE_SIZE slot. Values shorter than
/// STAT_VALUE_SIZE are placed at the start with zero-padding on the right.
/// This preserves little-endian byte ordering for fixed-size values,
/// so lexicographic comparison of slots matches comparison of the
/// original values.
pub fn value_to_stat_slot(value: &[u8]) -> [u8; STAT_VALUE_SIZE] {
    let mut slot = [0u8; STAT_VALUE_SIZE];
    let len = value.len().min(STAT_VALUE_SIZE);
    slot[..len].copy_from_slice(&value[..len]);
    slot
}

/// True when this column's fixed-width value is a signed integer in two's
/// complement (so a negative value's high byte has bit 7 set and an unsigned
/// byte comparison would rank it above every non-negative value). Temporal
/// and HLC columns are integer-backed (Date i32, Time/Timestamp/TimestampTz/
/// Interval i64, ps Timestamp/HLC i128) and are stamped from a signed instant,
/// so a pre-1970 / negative picosecond must sort below 1970. Unsigned integer
/// types keep the plain unsigned ordering.
pub fn stat_slot_is_signed(type_id: TypeId) -> bool {
    matches!(
        type_id,
        TypeId::Int8
            | TypeId::Int16
            | TypeId::Int32
            | TypeId::Int64
            | TypeId::Int128
            | TypeId::Decimal
            | TypeId::Date
            | TypeId::Time
            | TypeId::Timestamp
            | TypeId::TimestampTz
            | TypeId::Interval
            | TypeId::Hlc
    )
}

/// How a column's stat slot bytes order against one another.
///
/// A slot holds the value's raw little endian bytes, and three different
/// orders read those bytes: an unsigned integer is ordered by them
/// directly, a two's complement integer needs its sign bit inverted
/// first, and an IEEE float needs the whole word inverted when it is
/// negative because a float is further from zero the larger its unsigned
/// reading. Recording a float's bounds under the wrong one of these does
/// not merely lose pruning, it loses the value: for a column holding both
/// signs the unsigned extremes are the smallest non-negative and the most
/// negative, and the largest value is never written down at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotOrder {
    /// Byte order is value order
    Unsigned,
    /// Two's complement, so a negative sorts below every non-negative
    TwosComplement,
    /// IEEE 754, where the sign bit reverses the direction
    Ieee,
    /// A variable-length value, ordered by its bytes from the first, and
    /// held as at most a 32-byte prefix
    Lexicographic,
}

/// The order one column's stat slots compare in.
pub fn slot_order(type_id: TypeId) -> SlotOrder {
    if type_id.fixed_size().is_none() {
        return SlotOrder::Lexicographic;
    }
    match type_id {
        TypeId::Float32 | TypeId::Float64 => SlotOrder::Ieee,
        other if stat_slot_is_signed(other) => SlotOrder::TwosComplement,
        _ => SlotOrder::Unsigned,
    }
}

/// Upper bound slot for a variable-length value.
///
/// A slot holds at most 32 bytes, and a truncated prefix sorts below the
/// value it came from, so recording it as a maximum would be a bound the
/// data breaks. Rounding the prefix up restores it: incrementing the last
/// byte that can carry, and zeroing what follows, gives the smallest
/// 32-byte string strictly above every value with that prefix. A value
/// whose prefix is already all ones has nothing above it to round to, and
/// the all-ones slot is itself a valid bound for anything.
pub fn varlen_upper_slot(value: &[u8]) -> [u8; STAT_VALUE_SIZE] {
    let mut slot = value_to_stat_slot(value);
    if value.len() <= STAT_VALUE_SIZE {
        return slot;
    }
    for i in (0..STAT_VALUE_SIZE).rev() {
        if slot[i] != 0xFF {
            slot[i] += 1;
            for byte in slot[i + 1..].iter_mut() {
                *byte = 0;
            }
            return slot;
        }
    }
    [0xFF; STAT_VALUE_SIZE]
}

/// Compares two slots holding variable-length prefixes, first byte first
fn compare_lexicographic(
    a: &[u8; STAT_VALUE_SIZE],
    b: &[u8; STAT_VALUE_SIZE],
) -> std::cmp::Ordering {
    a[..].cmp(&b[..])
}

/// Order-preserving key for a `width`-byte IEEE float held little endian.
///
/// Negative floats have their sign bit set and grow away from zero, so
/// the whole word is inverted; non-negative floats keep their order and
/// are lifted above every negative by setting the sign bit. This is the
/// same total order `f64::total_cmp` defines.
fn ieee_key(bytes: &[u8], width: usize) -> u128 {
    let mut raw: u128 = 0;
    for i in (0..width).rev() {
        raw = (raw << 8) | bytes.get(i).copied().unwrap_or(0) as u128;
    }
    let sign = 1u128 << (8 * width - 1);
    if raw & sign != 0 {
        // Within the width, so the padding above it stays zero
        !raw & ((sign << 1) - 1)
    } else {
        raw | sign
    }
}

/// Compares two stat slots holding `width`-byte little-endian values.
/// `signed` selects two's complement ordering: when the operands have
/// different sign bits the negative one is the smaller, otherwise (same sign,
/// including two negatives) the unsigned little-endian comparison of the value
/// bytes is already the correct order. The slot stays raw value bytes so the
/// metadata-aggregate decode reads the true signed extremum back unchanged.
pub fn compare_stat_slots_typed(
    a: &[u8; STAT_VALUE_SIZE],
    b: &[u8; STAT_VALUE_SIZE],
    width: usize,
    order: SlotOrder,
) -> std::cmp::Ordering {
    if order == SlotOrder::Lexicographic {
        return compare_lexicographic(a, b);
    }
    if order == SlotOrder::Ieee && (width == 4 || width == 8) {
        return ieee_key(a, width).cmp(&ieee_key(b, width));
    }
    if order == SlotOrder::TwosComplement && width > 0 {
        let sign_byte = width - 1;
        let a_neg = a[sign_byte] & 0x80 != 0;
        let b_neg = b[sign_byte] & 0x80 != 0;
        if a_neg != b_neg {
            // The negative operand is the smaller one.
            return if a_neg {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
    }
    compare_stat_slots(a, b)
}

/// Orders a raw fixed-width value against an existing stat slot without
/// materializing a 32-byte slot for the value. A stat slot is the value bytes
/// left-aligned and zero-padded, so for `width <= STAT_VALUE_SIZE` comparing
/// the value's `width` bytes (zero-extended) against the slot's `width` bytes
/// is identical to comparing two slots. This lets `ColumnSegment::build`
/// compare every row's value to the running min/max and only build the
/// 32-byte slot on the rare row that actually moves a bound.
pub fn compare_value_to_slot(
    v: &[u8],
    slot: &[u8; STAT_VALUE_SIZE],
    width: usize,
    order: SlotOrder,
) -> std::cmp::Ordering {
    let vb = |i: usize| -> u8 { v.get(i).copied().unwrap_or(0) };
    if order == SlotOrder::Lexicographic {
        // The value's own prefix against the recorded one. Comparing
        // prefixes decides the values whenever they differ inside the
        // first 32 bytes, and a tie there is what the bounds treat
        // conservatively
        return compare_lexicographic(&value_to_stat_slot(v), slot);
    }
    if order == SlotOrder::Ieee && (width == 4 || width == 8) {
        return ieee_key(v, width).cmp(&ieee_key(slot, width));
    }
    if order == SlotOrder::TwosComplement && width > 0 {
        let sign_byte = width - 1;
        let a_neg = vb(sign_byte) & 0x80 != 0;
        let b_neg = slot[sign_byte] & 0x80 != 0;
        if a_neg != b_neg {
            return if a_neg {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
    }
    // Unsigned little-endian compare, most significant byte first. Bytes at or
    // above `width` are zero in both operands (slots are zero-padded), so only
    // the value bytes need to be compared.
    for i in (0..width.min(STAT_VALUE_SIZE)).rev() {
        match vb(i).cmp(&slot[i]) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

/// Orders two of one column's raw values against each other, under the same
/// order its stat slots compare in.
///
/// A segment build tracks its extremes as the values themselves and builds a
/// 32-byte slot only when a zone closes, so this is the comparison every row
/// runs through rather than the value-to-slot form. Both operands are one
/// column's cells, so for a fixed-width column they are `width` bytes each.
pub fn compare_values_ordered(
    a: &[u8],
    b: &[u8],
    width: usize,
    order: SlotOrder,
) -> std::cmp::Ordering {
    if order == SlotOrder::Lexicographic {
        // Byte order from the first byte, which is the order a
        // variable-length value's own bytes carry, and unlike the slot form
        // it compares the whole value rather than a 32-byte prefix
        return a.cmp(b);
    }
    if order == SlotOrder::Ieee && (width == 4 || width == 8) {
        return ieee_key(a, width).cmp(&ieee_key(b, width));
    }
    if order == SlotOrder::TwosComplement && width > 0 {
        let sign_byte = width - 1;
        let a_neg = a.get(sign_byte).copied().unwrap_or(0) & 0x80 != 0;
        let b_neg = b.get(sign_byte).copied().unwrap_or(0) & 0x80 != 0;
        if a_neg != b_neg {
            // The negative operand is the smaller one
            return if a_neg {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
    }
    compare_le_bytes(a, b)
}

/// Closes one zone: records its bounds as a zone map entry and folds them
/// into the segment's running extremes.
///
/// Folding the segment bounds out of the zone bounds is what keeps the row
/// loop to one comparison per bound. A zone with no non-null value records
/// an inverted pair, which no interval overlaps, so a pruner reads it as
/// admitting nothing
#[inline]
fn close_zone<'v>(
    zoneMaps: &mut Vec<ZoneMapEntry>,
    zoneMin: Option<(usize, &'v [u8])>,
    zoneMax: Option<(usize, &'v [u8])>,
    segmentMin: &mut Option<(usize, &'v [u8])>,
    segmentMax: &mut Option<(usize, &'v [u8])>,
    width: usize,
    order: SlotOrder,
) {
    zoneMaps.push(ZoneMapEntry {
        min_value: zoneMin.map_or([0xFF; STAT_VALUE_SIZE], |(_, v)| value_to_stat_slot(v)),
        // A value wider than a slot is held as a prefix, and a prefix sorts
        // below the value it came from, so the upper bound is that prefix
        // rounded up. A value that fits is its own bound
        max_value: zoneMax.map_or([0u8; STAT_VALUE_SIZE], |(_, v)| varlen_upper_slot(v)),
    });
    if let Some((row, value)) = zoneMin
        && segmentMin
            .is_none_or(|(_, cur)| compare_values_ordered(value, cur, width, order).is_lt())
    {
        *segmentMin = Some((row, value));
    }
    if let Some((row, value)) = zoneMax
        && segmentMax
            .is_none_or(|(_, cur)| compare_values_ordered(value, cur, width, order).is_gt())
    {
        *segmentMax = Some((row, value));
    }
}

/// Distinct value tracking for one segment build.
///
/// The bloom decision only needs to know whether the column holds more than
/// BLOOM_MIN_CARDINALITY distinct values, so the exact set stops one past
/// the threshold and the rest of the column costs nothing. A caller that
/// also publishes a distinct estimate asks for the sketch, and then the
/// exact set is keyed by the hash the sketch already computed, so a value is
/// hashed once instead of once for each structure.
struct DistinctTracker<'v> {
    /// Exact set while the caller wants no estimate, keyed by the bytes
    values: hashbrown::HashSet<&'v [u8]>,
    /// Whole-column estimate, boxed so a build that does not ask for one
    /// carries a pointer rather than the register array.
    ///
    /// It counts its own first distinct hashes exactly, out to the cap this
    /// tracker was built for, so both the bloom threshold and the encoding
    /// decision read one structure that the value was hashed into once
    sketch: Option<Box<DistinctSketch>>,
    /// Distinct values counted before the count stops changing a decision
    cap: usize,
    /// Set once more distinct values arrived than `cap`, tracked here only
    /// for the path that keeps no sketch
    saturated: bool,
}

impl<'v> DistinctTracker<'v> {
    fn new(sketch: bool, cap: usize) -> Self {
        Self {
            values: hashbrown::HashSet::new(),
            // The exact table is held to a fixed size rather than sized to
            // the cap. Counting exactly out to half the row count would
            // allocate and zero a quarter of a megabyte for a sixteen
            // thousand row column and a megabyte for a quarter million row
            // one, per column, to sharpen a threshold the registers already
            // decide the same way. What still needs an exact answer is the
            // bloom threshold at sixty four, which sits inside this
            sketch: sketch.then(|| {
                Box::new(DistinctSketch::with_exact_capacity(
                    cap.min(super::sketch::EXACT_CAPACITY),
                ))
            }),
            cap,
            saturated: false,
        }
    }

    #[inline]
    fn insert(&mut self, value: &'v [u8]) {
        match self.sketch.as_mut() {
            Some(sketch) => sketch.insert(value),
            None => {
                if !self.saturated {
                    self.values.insert(value);
                    self.saturated = self.values.len() > self.cap;
                }
            }
        }
    }

    /// Distinct values counted exactly, saturating one past `cap`.
    ///
    /// Under a sketch the count is over hashes rather than over the values
    /// themselves. Two distinct values sharing a 64-bit hash would be
    /// counted once, which no correctness decision rests on: the encoding
    /// this feeds picks a dictionary rather than a bit-packing, and a
    /// dictionary round-trips whatever the column holds
    #[inline]
    fn counted(&self) -> usize {
        match self.sketch.as_ref() {
            Some(sketch) => sketch.counted(),
            None => self.values.len(),
        }
    }

    /// Distinct values as a segment header records them, stopping one past
    /// the bloom threshold, which is the only thing that reads the field
    fn cardinality(&self) -> u64 {
        (self.counted() as u64).min(BLOOM_MIN_CARDINALITY + 1)
    }

    /// Distinct values as the encoding decision reads them: exact while the
    /// table holds every hash seen, estimated past it.
    ///
    /// The threshold this feeds sits at half the row count, far above where
    /// the exact table stops, and the registers carry about one percent of
    /// error there. A column near enough to the threshold for that to flip
    /// the answer stores about as well either way, and both encodings round
    /// trip whatever the column holds
    fn estimated(&self) -> usize {
        match self.sketch.as_ref() {
            Some(sketch) => sketch.estimate() as usize,
            None => self.values.len(),
        }
    }

    /// Whether the count passed the bloom threshold
    fn saturated(&self) -> bool {
        self.counted() as u64 > BLOOM_MIN_CARDINALITY
    }

    /// Whole-column distinct estimate, present when the caller asked for it
    fn estimate(&self) -> Option<u64> {
        self.sketch.as_ref().map(|sketch| sketch.estimate())
    }
}

/// Compares two stat slots as unsigned little-endian integers.
/// Returns Ordering::Less, Equal, or Greater.
/// For LE values, comparison starts from the most significant byte (highest index
/// with non-zero content) and works down, matching numeric ordering.
pub fn compare_stat_slots(
    a: &[u8; STAT_VALUE_SIZE],
    b: &[u8; STAT_VALUE_SIZE],
) -> std::cmp::Ordering {
    // Compare from highest byte index down (most significant byte first for LE).
    for i in (0..STAT_VALUE_SIZE).rev() {
        match a[i].cmp(&b[i]) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

/// Widest cell the row loop folds into a single comparison word. Every
/// fixed-size type but `Inet` and `Cidr`, which are eighteen bytes, fits
const WORD_KEY_MAX_WIDTH: usize = 16;

/// Reads a fixed-width cell as an unsigned little-endian integer.
///
/// Two cells of one column are the same width, so comparing the words is
/// comparing the cells, and a comparison becomes one integer compare rather
/// than a walk from the last byte down.
///
/// The widths a fixed-size column is made of are resolved to a single load
/// each. Staging every cell through a sixteen-byte buffer instead measured
/// twice the cost at every width, because a machine integer's worth of
/// bytes goes through a zeroing and a copy to be read back as one word
#[inline(always)]
fn le_word(value: &[u8]) -> u128 {
    match value.len() {
        8 => match <[u8; 8]>::try_from(value) {
            Ok(bytes) => u64::from_le_bytes(bytes) as u128,
            Err(_) => 0,
        },
        4 => match <[u8; 4]>::try_from(value) {
            Ok(bytes) => u32::from_le_bytes(bytes) as u128,
            Err(_) => 0,
        },
        16 => match <[u8; 16]>::try_from(value) {
            Ok(bytes) => u128::from_le_bytes(bytes),
            Err(_) => 0,
        },
        2 => match <[u8; 2]>::try_from(value) {
            Ok(bytes) => u16::from_le_bytes(bytes) as u128,
            Err(_) => 0,
        },
        1 => value.first().map_or(0, |&byte| byte as u128),
        _ => {
            let mut word = [0u8; WORD_KEY_MAX_WIDTH];
            let span = value.len().min(WORD_KEY_MAX_WIDTH);
            word[..span].copy_from_slice(&value[..span]);
            u128::from_le_bytes(word)
        }
    }
}

/// Folds a cell's little-endian word into the order that column's cells
/// compare in.
///
/// Comparing two folded words answers what [`compare_values_ordered`]
/// answers for the cells they came from, for every width this is called at
#[inline(always)]
fn ordered_word(raw: u128, width: usize, order: SlotOrder) -> u128 {
    match order {
        SlotOrder::Unsigned => raw,
        // A set sign bit means negative, and lifting it clears of every
        // non-negative value puts the two groups the right way round while
        // leaving the order inside each group alone
        SlotOrder::TwosComplement => raw ^ (1u128 << (width * 8 - 1)),
        SlotOrder::Ieee if width == 4 || width == 8 => {
            let sign = 1u128 << (width * 8 - 1);
            if raw & sign != 0 {
                !raw & ((sign << 1) - 1)
            } else {
                raw | sign
            }
        }
        // A float at a width IEEE does not define falls back to the plain
        // value order, which is what the general comparator does with it
        SlotOrder::Ieee => raw,
        // Byte order from the first byte, which reversing a left-aligned
        // little-endian word moves into the top of the word
        SlotOrder::Lexicographic => raw.swap_bytes(),
    }
}

/// Orders two cells, through their comparison words when the column's width
/// folds into one and through the general comparator when it does not.
///
/// `word_keys` is fixed for a whole segment, so the branch resolves the
/// same way on every row of a build
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cell_is_less(
    candidate: u128,
    candidate_bytes: &[u8],
    current: u128,
    current_bytes: &[u8],
    width: usize,
    order: SlotOrder,
    word_keys: bool,
) -> bool {
    if word_keys {
        candidate < current
    } else {
        compare_values_ordered(candidate_bytes, current_bytes, width, order).is_lt()
    }
}

/// Compares two equal-length byte slices as unsigned little-endian integers.
/// For LE values, the last byte is the most significant, so comparison
/// starts from the highest index and works down.
pub fn compare_le_bytes(a: &[u8], b: &[u8]) -> std::cmp::Ordering {
    debug_assert_eq!(a.len(), b.len());
    for i in (0..a.len()).rev() {
        match a[i].cmp(&b[i]) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

impl ColumnSegment {
    /// Builds a ColumnSegment from raw column values, letting the segment's
    /// own cardinality decide whether a bloom filter is worth its bytes.
    ///
    /// `column_id` - ordinal position of this column in the table schema.
    /// `type_id` - data type of the column, used for encoding selection.
    /// `value_size` - byte width of each value (fixed-size types only).
    /// `values` - row values, where None represents a null.
    pub fn build(
        columnId: u32,
        typeId: TypeId,
        valueSize: usize,
        values: &[Option<&[u8]>],
    ) -> Result<Self> {
        Self::build_with_options(
            columnId,
            typeId,
            valueSize,
            values,
            SegmentOptions::default(),
        )
    }

    /// Builds a ColumnSegment with the caller choosing the bloom policy and
    /// how the encoding is picked.
    pub fn build_with_options(
        columnId: u32,
        typeId: TypeId,
        valueSize: usize,
        values: &[Option<&[u8]>],
        options: SegmentOptions,
    ) -> Result<Self> {
        let rowCount = values.len();
        if rowCount == 0 {
            return Err(ZyronError::EncodingFailed(
                "cannot build segment from zero rows".to_string(),
            ));
        }

        if valueSize == 0 {
            return Self::build_varlen(columnId, typeId, values, options);
        }

        // Pack the cells into the buffer the encoder reads.
        //
        // This input holds a pointer per cell, so the walk that follows
        // would chase one for every value it bounds, hashes and compares.
        // Packing first costs one pass and leaves every later pass reading
        // flat memory. A caller already holding its column packed skips
        // this entirely through `build_packed`
        //
        // SAFETY: the loop below writes every one of `buf_len` slots, null
        // slots explicitly zeroed and non-null slots copied, before anything
        // reads rawData. Zeroing up front would memset the whole column
        // buffer only to overwrite it
        let buf_len = rowCount * valueSize;
        #[allow(clippy::uninit_vec)]
        let mut rawData: Vec<u8> = {
            let mut v = Vec::with_capacity(buf_len);
            unsafe { v.set_len(buf_len) };
            v
        };
        let mut nullBitmap: Vec<u8> = Vec::new();
        let mut nullCount = 0u64;
        for (i, val) in values.iter().enumerate() {
            let start = i * valueSize;
            match val {
                None => {
                    nullCount += 1;
                    if nullBitmap.is_empty() {
                        nullBitmap = vec![0u8; rowCount.div_ceil(8)];
                    }
                    nullBitmap[i / 8] |= 1 << (i % 8);
                    // The buffer was allocated uninitialized, and encoders
                    // treat a null slot as a deterministic zero placeholder
                    rawData[start..start + valueSize].fill(0);
                }
                Some(v) => {
                    if v.len() != valueSize {
                        // A non-null value whose width does not match the
                        // fixed column value size cannot be packed into its
                        // slot, fail instead of zero-filling and corrupting
                        // the value
                        return Err(ZyronError::EncodingFailed(format!(
                            "non-null value at row {} has length {} expected {}",
                            i,
                            v.len(),
                            valueSize
                        )));
                    }
                    rawData[start..start + valueSize].copy_from_slice(v);
                }
            }
        }

        Self::build_from_packed(
            columnId, typeId, valueSize, &rawData, nullBitmap, nullCount, rowCount, options,
        )
    }

    /// Builds a fixed-width segment from a buffer already laid out the way
    /// the encoder reads it: every cell at `valueSize` bytes, null slots
    /// zero-filled, and a bitmap naming which those are.
    ///
    /// A caller holding its column in that shape hands it straight in and
    /// the segment is built without copying it. The form that takes a
    /// pointer per cell packs a buffer like this first, so both arrive at
    /// the same pass and only one of them pays to build it.
    ///
    /// An empty `nulls` means the column has none
    pub fn build_packed(
        columnId: u32,
        typeId: TypeId,
        valueSize: usize,
        values: &[u8],
        nulls: &[u8],
        rowCount: usize,
        options: SegmentOptions,
    ) -> Result<Self> {
        if rowCount == 0 {
            return Err(ZyronError::EncodingFailed(
                "cannot build segment from zero rows".to_string(),
            ));
        }
        if valueSize == 0 {
            return Err(ZyronError::EncodingFailed(
                "a packed segment needs a fixed value width".to_string(),
            ));
        }
        let span = rowCount * valueSize;
        if values.len() < span {
            return Err(ZyronError::EncodingFailed(format!(
                "packed buffer holds {} bytes, {} rows at {} bytes need {}",
                values.len(),
                rowCount,
                valueSize,
                span
            )));
        }
        let bitmapLen = rowCount.div_ceil(8);
        let mut nullCount = 0u64;
        for row in 0..rowCount {
            if nulls
                .get(row / 8)
                .is_some_and(|byte| byte & (1 << (row % 8)) != 0)
            {
                nullCount += 1;
            }
        }
        // A null slot carries a deterministic zero so that an all-null
        // column encodes as one repeated value and two segments over the
        // same rows produce the same bytes. A caller that leaves data in a
        // slot it called null would get a file whose nulls decode to
        // whatever was there, so the contract is checked rather than just
        // written down
        debug_assert!(
            (0..rowCount).all(|row| {
                !nulls
                    .get(row / 8)
                    .is_some_and(|byte| byte & (1 << (row % 8)) != 0)
                    || values[row * valueSize..(row + 1) * valueSize].iter().all(|&b| b == 0)
            }),
            "a null slot in the packed buffer is not zero-filled"
        );
        let nullBitmap = if nullCount == 0 {
            Vec::new()
        } else {
            let mut bitmap = vec![0u8; bitmapLen];
            let copied = bitmapLen.min(nulls.len());
            bitmap[..copied].copy_from_slice(&nulls[..copied]);
            // Bits past the row count belong to no row and would read as
            // nulls the column does not have
            if rowCount % 8 != 0
                && let Some(last) = bitmap.last_mut()
            {
                *last &= (1u8 << (rowCount % 8)) - 1;
            }
            bitmap
        };
        Self::build_from_packed(
            columnId,
            typeId,
            valueSize,
            &values[..span],
            nullBitmap,
            nullCount,
            rowCount,
            options,
        )
    }

    /// The one pass both fixed-width entry points arrive at, over a buffer
    /// laid out the way the encoder reads it.
    ///
    /// Computes in a single walk:
    ///   - per-zone min/max emitted at every ZONE_MAP_BATCH_SIZE boundary,
    ///     and the segment's own min/max folded out of those at each zone
    ///     boundary rather than compared against on every row
    ///   - the rows those two extremes sit at, so a caller that needs the
    ///     value at its full width reads it back instead of running its own
    ///     pass over the same column
    ///   - the sorted flag and the run count, which is what decides whether
    ///     the column is run-length shaped
    ///   - distinct tracking, counted far enough for both the bloom
    ///     threshold and the dictionary decision, plus the whole-column
    ///     sketch when the caller asked for an estimate
    ///
    /// Bounds are carried as the values themselves, so a 32-byte slot is
    /// built twice per zone rather than on every row that moves a bound
    #[allow(clippy::too_many_arguments)]
    fn build_from_packed(
        columnId: u32,
        typeId: TypeId,
        valueSize: usize,
        rawData: &[u8],
        nullBitmap: Vec<u8>,
        nullCount: u64,
        rowCount: usize,
        options: SegmentOptions,
    ) -> Result<Self> {
        let rawSize = (rowCount * valueSize) as u64;

        // Two's complement ordering for signed columns so a negative value
        // (incl. a pre-1970 picosecond timestamp) sorts below zero in the
        // segment min/max instead of above every positive under an unsigned
        // byte compare.
        let statOrder = slot_order(typeId);

        // Cells this narrow fold into one comparison word, which is every
        // fixed-size type but the two that are eighteen bytes wide. Resolved
        // once here rather than asked per row
        let wordKeys = valueSize <= WORD_KEY_MAX_WIDTH;
        // A column with no nulls skips the bitmap read on every row
        let hasNulls = nullCount > 0;

        let batchSize = ZONE_MAP_BATCH_SIZE as usize;
        let zoneCount = rowCount.div_ceil(batchSize);
        let mut zoneMaps: Vec<ZoneMapEntry> = Vec::with_capacity(zoneCount);

        // Counted far enough for both readers of the count: the bloom
        // threshold, and the dictionary decision that selection would
        // otherwise walk the column a second time to make
        let mut distinct = DistinctTracker::new(
            options.distinct_sketch,
            cardinality_cap(rowCount).max(BLOOM_MIN_CARDINALITY as usize),
        );
        // Adjacent non-null values that differ, plus one, which is what
        // decides whether the column is run-length shaped
        let mut runCount = 1usize;
        let mut segmentMin: Option<(usize, &[u8])> = None;
        let mut segmentMax: Option<(usize, &[u8])> = None;
        let mut isSorted = true;
        let mut prevRaw: Option<&[u8]> = None;
        // Unsigned word of the previous non-null cell, which is the order
        // the sorted flag records
        let mut prevWord = 0u128;

        // Zone bounds carry the comparison word beside the value, so a row
        // that does not move a bound costs one integer compare per bound
        let mut zoneMin: Option<(usize, &[u8], u128)> = None;
        let mut zoneMax: Option<(usize, &[u8], u128)> = None;
        let mut zoneEnd = batchSize;

        for i in 0..rowCount {
            if i == zoneEnd {
                close_zone(
                    &mut zoneMaps,
                    zoneMin.map(|(row, value, _)| (row, value)),
                    zoneMax.map(|(row, value, _)| (row, value)),
                    &mut segmentMin,
                    &mut segmentMax,
                    valueSize,
                    statOrder,
                );
                zoneMin = None;
                zoneMax = None;
                zoneEnd += batchSize;
            }

            if hasNulls
                && nullBitmap
                    .get(i / 8)
                    .is_some_and(|byte| byte & (1 << (i % 8)) != 0)
            {
                continue;
            }
            let v = &rawData[i * valueSize..(i + 1) * valueSize];
            distinct.insert(v);

            // One fold of the cell serves both bounds and the sorted flag.
            // The unsigned word is the order the flag records, and the
            // ordered word is the order the bounds compare in
            let rawWord = if wordKeys { le_word(v) } else { 0 };
            let ordWord = if wordKeys {
                ordered_word(rawWord, valueSize, statOrder)
            } else {
                0
            };

            // Only the zone this row lands in has to be beaten. The segment
            // bounds are folded out of the zone bounds when the zone closes,
            // which is one comparison per bound per row rather than two
            if zoneMin.is_none_or(|(_, cur, curWord)| {
                cell_is_less(ordWord, v, curWord, cur, valueSize, statOrder, wordKeys)
            }) {
                zoneMin = Some((i, v, ordWord));
            }
            if zoneMax.is_none_or(|(_, cur, curWord)| {
                cell_is_less(curWord, cur, ordWord, v, valueSize, statOrder, wordKeys)
            }) {
                zoneMax = Some((i, v, ordWord));
            }

            if let Some(prev) = prevRaw {
                // The fold is a zero-extended copy of the cell, so two words
                // are equal exactly when the cells are and the run count
                // stays exact
                let differs = if wordKeys {
                    rawWord != prevWord
                } else {
                    v != prev
                };
                if differs {
                    runCount += 1;
                }
                if isSorted {
                    let descended = if wordKeys {
                        rawWord < prevWord
                    } else {
                        compare_le_bytes(v, prev) == std::cmp::Ordering::Less
                    };
                    if descended {
                        isSorted = false;
                    }
                }
            }
            prevRaw = Some(v);
            prevWord = rawWord;
        }
        // Close the final zone, which may be partially filled
        if zoneMaps.len() < zoneCount {
            close_zone(
                &mut zoneMaps,
                zoneMin.map(|(row, value, _)| (row, value)),
                zoneMax.map(|(row, value, _)| (row, value)),
                &mut segmentMin,
                &mut segmentMax,
                valueSize,
                statOrder,
            );
        }

        let cardinality = distinct.cardinality();
        let distinctCapped = distinct.saturated();
        let minValue = segmentMin.map_or([0u8; STAT_VALUE_SIZE], |(_, v)| value_to_stat_slot(v));
        let maxValue = segmentMax.map_or([0u8; STAT_VALUE_SIZE], |(_, v)| varlen_upper_slot(v));

        // Selection is handed the buffer the fused pass just packed and the
        // statistics that same pass gathered, and hands back the bytes it
        // produced while deciding. Under exact selection the trial is a full
        // encode of the column, so taking its output is the difference
        // between encoding this column once and encoding it twice, and
        // handing the statistics in is the difference between reading the
        // column once and reading it twice
        //
        // A column is all one value when every cell is null, or when no cell
        // is null and no adjacent pair differs. Deriving it from the run
        // count rather than from the distinct count keeps it exact under a
        // set that is keyed by hashes
        let allIdentical =
            nullCount as usize == rowCount || (nullCount == 0 && runCount == 1);
        let stats = ColumnSampleStats {
            cardinality: distinct.estimated(),
            run_count: runCount,
            all_identical: allIdentical,
        };
        let choice = select_encoding_prepared(
            typeId,
            rowCount,
            stats,
            &rawData,
            valueSize,
            options.exact_encoding,
        );
        let encodingType = choice.encoding;
        let encodedData = match choice.encoded {
            Some(bytes) => bytes,
            None => create_encoding(encodingType).encode(&rawData, rowCount, valueSize)?,
        };
        let encodedSize = encodedData.len() as u64;

        // Build bloom filter when the policy asks for one, this is the only
        // remaining pass over values since the filter sizing depends on the
        // cardinality decision computed in the fused pass above
        let bloomFilter = if options.bloom.builds_bloom(cardinality, encodingType) {
            let bloom_size_hint = if distinctCapped {
                rowCount as u64
            } else {
                cardinality
            };
            let mut filter = BloomFilter::new(bloom_size_hint);
            for row in 0..rowCount {
                if hasNulls
                    && nullBitmap
                        .get(row / 8)
                        .is_some_and(|byte| byte & (1 << (row % 8)) != 0)
                {
                    continue;
                }
                filter.insert(&rawData[row * valueSize..(row + 1) * valueSize]);
            }
            Some(filter)
        } else {
            None
        };

        let bloomFilterSize = bloomFilter
            .as_ref()
            .map_or(0, |bf| bf.on_disk_size() as u32);

        let header = SegmentHeader {
            column_id: columnId,
            encoding_type: encodingType,
            raw_size: rawSize,
            encoded_size: encodedSize,
            null_count: nullCount,
            cardinality,
            min_value: minValue,
            max_value: maxValue,
            data_checksum: zyron_common::hash32(&encodedData),
            header_crc: 0,
            // The writer lays a segment out as header, bloom, zone maps,
            // null bitmap, data, so the bloom starts right after the header
            bloom_filter_offset: if bloomFilterSize > 0 {
                SEGMENT_HEADER_SIZE as u64
            } else {
                0
            },
            bloom_filter_size: bloomFilterSize,
            is_sorted: isSorted,
        };

        Ok(Self {
            header,
            bloom_filter: bloomFilter,
            zone_maps: zoneMaps,
            encoded_data: encodedData,
            null_bitmap: nullBitmap,
            min_row: segmentMin.map(|(row, _)| row),
            max_row: segmentMax.map(|(row, _)| row),
            ndv: distinct.estimate(),
        })
    }

    /// Builds a variable-length column segment. Values are stored in the
    /// canonical variable-length buffer (a u32 offset array plus a values
    /// blob).
    ///
    /// Bounds are tracked on the whole values and turned into slots when a
    /// zone closes: the minimum keeps its left-aligned, zero-padded byte
    /// prefix into STAT_VALUE_SIZE, and the maximum's prefix is rounded up so
    /// it stays above every value it covers. The same prefix transform is
    /// applied to predicate literals at prune time, so a shared prefix only
    /// ever widens a zone (a conservative non-skip), never a false skip. The
    /// null bitmap is authoritative, so an empty string and a null stay
    /// distinct.
    fn build_varlen(
        columnId: u32,
        typeId: TypeId,
        values: &[Option<&[u8]>],
        options: SegmentOptions,
    ) -> Result<Self> {
        let rowCount = values.len();
        let batchSize = ZONE_MAP_BATCH_SIZE as usize;
        let zoneCount = rowCount.div_ceil(batchSize);
        let mut zoneMaps: Vec<ZoneMapEntry> = Vec::with_capacity(zoneCount);

        let mut nullCount = 0u64;
        let mut nullBitmap: Vec<u8> = Vec::new();
        // Variable-length selection runs its own statistics over the packed
        // buffer, so the only reader of this count is the bloom threshold
        let mut distinct =
            DistinctTracker::new(options.distinct_sketch, BLOOM_MIN_CARDINALITY as usize);
        let mut segmentMin: Option<(usize, &[u8])> = None;
        let mut segmentMax: Option<(usize, &[u8])> = None;
        let mut isSorted = true;
        let mut prevRaw: Option<&[u8]> = None;

        let mut zoneMin: Option<(usize, &[u8])> = None;
        let mut zoneMax: Option<(usize, &[u8])> = None;
        let mut zoneEnd = batchSize;

        for (i, val) in values.iter().enumerate() {
            if i == zoneEnd {
                close_zone(
                    &mut zoneMaps,
                    zoneMin,
                    zoneMax,
                    &mut segmentMin,
                    &mut segmentMax,
                    0,
                    SlotOrder::Lexicographic,
                );
                zoneMin = None;
                zoneMax = None;
                zoneEnd += batchSize;
            }

            match val {
                None => {
                    nullCount += 1;
                    if nullBitmap.is_empty() {
                        nullBitmap = vec![0u8; rowCount.div_ceil(8)];
                    }
                    nullBitmap[i / 8] |= 1 << (i % 8);
                }
                Some(v) => {
                    let v: &[u8] = v;
                    distinct.insert(v);
                    // A variable-length column orders by its bytes from the
                    // first. Comparing whole values rather than their padded
                    // prefixes decides two values that agree over the first
                    // 32 bytes, so the bounds a zone closes with are the
                    // zone's real extremes and not merely a pair its prefixes
                    // tie for
                    if zoneMin.is_none_or(|(_, cur)| v < cur) {
                        zoneMin = Some((i, v));
                    }
                    if zoneMax.is_none_or(|(_, cur)| v > cur) {
                        zoneMax = Some((i, v));
                    }

                    if isSorted
                        && let Some(prev) = prevRaw
                        && v.cmp(prev) == std::cmp::Ordering::Less
                    {
                        isSorted = false;
                    }
                    prevRaw = Some(v);
                }
            }
        }
        if zoneMaps.len() < zoneCount {
            close_zone(
                &mut zoneMaps,
                zoneMin,
                zoneMax,
                &mut segmentMin,
                &mut segmentMax,
                0,
                SlotOrder::Lexicographic,
            );
        }

        let cardinality = distinct.cardinality();
        let distinctCapped = distinct.saturated();
        let minValue = segmentMin.map_or([0u8; STAT_VALUE_SIZE], |(_, v)| value_to_stat_slot(v));
        let maxValue = segmentMax.map_or([0u8; STAT_VALUE_SIZE], |(_, v)| varlen_upper_slot(v));

        // The canonical variable-length buffer addresses the values blob with
        // a u32 cumulative offset array. A blob (or row count) past u32 would
        // wrap an offset and silently return wrong slices on decode, so the
        // fold is rejected here and must split the column into smaller
        // segments instead of corrupting. Segments are already row-bounded by
        // the compaction max-rows policy, so this is a guard, not a normal
        // path.
        let blobLen: usize = values.iter().map(|v| v.map_or(0, |b| b.len())).sum();
        if blobLen > u32::MAX as usize {
            return Err(ZyronError::EncodingFailed(format!(
                "variable-length segment blob {} bytes exceeds the {}-byte u32 \
                 offset limit, the fold must split this column into smaller segments",
                blobLen,
                u32::MAX
            )));
        }
        if rowCount > u32::MAX as usize {
            return Err(ZyronError::EncodingFailed(format!(
                "variable-length segment row count {} exceeds the u32 limit",
                rowCount
            )));
        }

        let rawData = varlen_pack(values);
        let rawSize = rawData.len() as u64;

        let encodingType = select_encoding_varlen(typeId, values);
        let encoder = create_encoding(encodingType);
        let encodedData = encoder.encode(&rawData, rowCount, 0)?;
        let encodedSize = encodedData.len() as u64;

        let bloomFilter = if options.bloom.builds_bloom(cardinality, encodingType) {
            let bloom_size_hint = if distinctCapped {
                rowCount as u64
            } else {
                cardinality
            };
            let mut filter = BloomFilter::new(bloom_size_hint);
            for v in values.iter().flatten() {
                filter.insert(v);
            }
            Some(filter)
        } else {
            None
        };

        let bloomFilterSize = bloomFilter
            .as_ref()
            .map_or(0, |bf| bf.on_disk_size() as u32);

        let header = SegmentHeader {
            column_id: columnId,
            encoding_type: encodingType,
            raw_size: rawSize,
            encoded_size: encodedSize,
            null_count: nullCount,
            cardinality,
            min_value: minValue,
            max_value: maxValue,
            data_checksum: zyron_common::hash32(&encodedData),
            header_crc: 0,
            // The writer lays a segment out as header, bloom, zone maps,
            // null bitmap, data, so the bloom starts right after the header
            bloom_filter_offset: if bloomFilterSize > 0 {
                SEGMENT_HEADER_SIZE as u64
            } else {
                0
            },
            bloom_filter_size: bloomFilterSize,
            is_sorted: isSorted,
        };

        Ok(Self {
            header,
            bloom_filter: bloomFilter,
            zone_maps: zoneMaps,
            encoded_data: encodedData,
            null_bitmap: nullBitmap,
            min_row: segmentMin.map(|(row, _)| row),
            max_row: segmentMax.map(|(row, _)| row),
            ndv: distinct.estimate(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row loop compares cells through a folded word rather than
    /// through the general comparator, so the two have to answer the same
    /// question the same way at every width and under every order a
    /// fixed-size column can carry.
    ///
    /// Both the fixed bit patterns that sit on the boundaries and a spread
    /// of random pairs, because the sign bit, the all-ones value and zero
    /// are exactly where an order-preserving fold goes wrong
    #[test]
    fn folded_words_order_cells_the_way_the_comparator_does() {
        let orders = [
            SlotOrder::Unsigned,
            SlotOrder::TwosComplement,
            SlotOrder::Ieee,
            SlotOrder::Lexicographic,
        ];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for width in 1..=WORD_KEY_MAX_WIDTH {
            // Boundary patterns first, then random pairs over the same width
            let mut samples: Vec<Vec<u8>> = vec![
                vec![0x00; width],
                vec![0xFF; width],
                vec![0x80; width],
                vec![0x01; width],
            ];
            {
                let mut high = vec![0u8; width];
                high[width - 1] = 0x80;
                samples.push(high);
                let mut low = vec![0u8; width];
                low[0] = 0x01;
                samples.push(low);
            }
            for _ in 0..64 {
                let mut value = vec![0u8; width];
                for byte in value.iter_mut() {
                    *byte = (next() & 0xFF) as u8;
                }
                samples.push(value);
            }

            for order in orders {
                for a in &samples {
                    for b in &samples {
                        let expected = compare_values_ordered(a, b, width, order);
                        let folded = ordered_word(le_word(a), width, order)
                            .cmp(&ordered_word(le_word(b), width, order));
                        assert_eq!(
                            folded, expected,
                            "width {width} order {order:?} disagreed on {a:?} vs {b:?}"
                        );
                    }
                }
            }

            // And the unsigned word, which is the order the sorted flag is
            // recorded under
            for a in &samples {
                for b in &samples {
                    assert_eq!(
                        le_word(a).cmp(&le_word(b)),
                        compare_le_bytes(a, b),
                        "width {width} unsigned fold disagreed on {a:?} vs {b:?}"
                    );
                }
            }
        }
    }

    /// The packing pass gathers the statistics selection reads instead of
    /// letting selection walk the column again to rebuild them, so the two
    /// have to settle on the same encoding for every shape a column takes.
    ///
    /// `select_encoding_packed` still computes them from the cells, which
    /// makes it an independent oracle rather than a restatement of the
    /// folded path
    #[test]
    fn folded_statistics_choose_what_a_second_walk_chose() {
        fn packed_buffer(values: &[Option<&[u8]>], width: usize) -> Vec<u8> {
            let mut raw = vec![0u8; values.len() * width];
            for (i, value) in values.iter().enumerate() {
                if let Some(v) = value {
                    raw[i * width..i * width + width].copy_from_slice(v);
                }
            }
            raw
        }

        let rows = 4_096usize;
        let mut shapes: Vec<(&str, TypeId, usize, Vec<Option<Vec<u8>>>)> = Vec::new();

        // One repeated value, which is the constant case
        shapes.push((
            "all identical",
            TypeId::Int64,
            8,
            vec![Some(7i64.to_le_bytes().to_vec()); rows],
        ));
        // Every cell null, which is also constant
        shapes.push(("all null", TypeId::Int64, 8, vec![None; rows]));
        // One distinct value and a single null, which is not constant
        let mut one_and_a_null = vec![Some(7i64.to_le_bytes().to_vec()); rows];
        one_and_a_null[rows / 2] = None;
        shapes.push((
            "one value with a null",
            TypeId::Int64,
            8,
            one_and_a_null,
        ));
        shapes.push((
            "boolean",
            TypeId::Boolean,
            1,
            (0..rows).map(|r| Some(vec![(r % 2) as u8])).collect(),
        ));
        // Sparse enough for a dictionary
        shapes.push((
            "low cardinality",
            TypeId::Int64,
            8,
            (0..rows)
                .map(|r| Some(((r % 17) as i64).to_le_bytes().to_vec()))
                .collect(),
        ));
        // Long runs, which is the run-length shape
        shapes.push((
            "long runs",
            TypeId::Int64,
            8,
            (0..rows)
                .map(|r| Some(((r / 512) as i64).to_le_bytes().to_vec()))
                .collect(),
        ));
        // Distinct and ascending, which trial encoding decides
        shapes.push((
            "ascending distinct",
            TypeId::Int64,
            8,
            (0..rows).map(|r| Some((r as i64).to_le_bytes().to_vec())).collect(),
        ));
        // Distinct and scattered, with nulls through it
        shapes.push((
            "scattered with nulls",
            TypeId::Int64,
            8,
            (0..rows)
                .map(|r| {
                    if r % 23 == 0 {
                        None
                    } else {
                        Some(((r as i64).wrapping_mul(6_364_136_223_846_793_005)).to_le_bytes().to_vec())
                    }
                })
                .collect(),
        ));
        // Sitting exactly on the cardinality boundary the dictionary
        // decision compares against
        shapes.push((
            "half the rows distinct",
            TypeId::Int64,
            8,
            (0..rows)
                .map(|r| Some(((r / 2) as i64).to_le_bytes().to_vec()))
                .collect(),
        ));
        shapes.push((
            "one over half the rows distinct",
            TypeId::Int64,
            8,
            (0..rows)
                .map(|r| {
                    let value = if r < rows / 2 + 2 { r } else { r % (rows / 2 + 2) };
                    Some((value as i64).to_le_bytes().to_vec())
                })
                .collect(),
        ));
        shapes.push((
            "float column",
            TypeId::Float64,
            8,
            (0..rows)
                .map(|r| Some((r as f64 * 1.5).to_le_bytes().to_vec()))
                .collect(),
        ));
        shapes.push((
            "wide cells past one word",
            TypeId::Inet,
            18,
            (0..rows)
                .map(|r| {
                    let mut cell = vec![0u8; 18];
                    cell[..4].copy_from_slice(&(r as u32).to_le_bytes());
                    Some(cell)
                })
                .collect(),
        ));

        for (name, type_id, width, cells) in shapes {
            let views: Vec<Option<&[u8]>> = cells.iter().map(|c| c.as_deref()).collect();
            let raw = packed_buffer(&views, width);
            for exact in [true, false] {
                let expected =
                    crate::encoding::select_encoding_packed(type_id, &views, &raw, width, exact);
                let segment = ColumnSegment::build_with_options(
                    0,
                    type_id,
                    width,
                    &views,
                    SegmentOptions {
                        exact_encoding: exact,
                        ..SegmentOptions::default()
                    },
                )
                .expect("segment builds");
                assert_eq!(
                    segment.header.encoding_type, expected.encoding,
                    "{name} chose differently with exact_encoding {exact}"
                );
            }
        }
    }

    /// A caller holding its column packed hands the buffer straight in,
    /// and one that holds a pointer per cell packs a buffer first. Both
    /// arrive at the same pass, so both have to produce the same segment
    /// down to the encoded bytes, the zone maps and the bloom
    #[test]
    fn a_packed_build_matches_the_build_that_packs_for_itself() {
        let rows = 3_000usize;
        let shapes: Vec<(&str, TypeId, usize, Vec<Option<Vec<u8>>>)> = vec![
            (
                "ascending distinct",
                TypeId::Int64,
                8,
                (0..rows).map(|r| Some((r as i64).to_le_bytes().to_vec())).collect(),
            ),
            (
                "with nulls",
                TypeId::Int64,
                8,
                (0..rows)
                    .map(|r| (r % 7 != 0).then(|| (r as i64).to_le_bytes().to_vec()))
                    .collect(),
            ),
            ("all null", TypeId::Int64, 8, vec![None; rows]),
            (
                "one repeated value",
                TypeId::Int64,
                8,
                vec![Some(42i64.to_le_bytes().to_vec()); rows],
            ),
            (
                "low cardinality",
                TypeId::Int64,
                8,
                (0..rows)
                    .map(|r| Some(((r % 11) as i64).to_le_bytes().to_vec()))
                    .collect(),
            ),
            (
                "signed across zero",
                TypeId::Int64,
                8,
                (0..rows)
                    .map(|r| Some((r as i64 - (rows as i64 / 2)).to_le_bytes().to_vec()))
                    .collect(),
            ),
            (
                "floats",
                TypeId::Float64,
                8,
                (0..rows)
                    .map(|r| Some(((r as f64) - 1500.0).to_le_bytes().to_vec()))
                    .collect(),
            ),
            (
                "narrow cells",
                TypeId::Int16,
                2,
                (0..rows).map(|r| Some((r as i16).to_le_bytes().to_vec())).collect(),
            ),
            (
                "cells wider than one word",
                TypeId::Inet,
                18,
                (0..rows)
                    .map(|r| {
                        let mut cell = vec![0u8; 18];
                        cell[..4].copy_from_slice(&(r as u32).to_le_bytes());
                        Some(cell)
                    })
                    .collect(),
            ),
        ];

        for (name, type_id, width, cells) in shapes {
            let views: Vec<Option<&[u8]>> = cells.iter().map(|c| c.as_deref()).collect();
            // The buffer a packed caller holds: cells at their stride with
            // null slots zeroed, and a bitmap naming the nulls
            let mut packed = vec![0u8; cells.len() * width];
            let mut nulls = vec![0u8; cells.len().div_ceil(8)];
            for (row, cell) in cells.iter().enumerate() {
                match cell {
                    Some(value) => {
                        packed[row * width..(row + 1) * width].copy_from_slice(value)
                    }
                    None => nulls[row / 8] |= 1 << (row % 8),
                }
            }

            for sketch in [false, true] {
                let options = SegmentOptions {
                    distinct_sketch: sketch,
                    exact_encoding: true,
                    ..SegmentOptions::default()
                };
                let from_views =
                    ColumnSegment::build_with_options(3, type_id, width, &views, options)
                        .expect("views build");
                let from_packed = ColumnSegment::build_packed(
                    3,
                    type_id,
                    width,
                    &packed,
                    &nulls,
                    cells.len(),
                    options,
                )
                .expect("packed build");

                let tag = format!("{name} with sketch {sketch}");
                assert_eq!(
                    from_packed.header.encoding_type, from_views.header.encoding_type,
                    "{tag}: encoding"
                );
                assert_eq!(
                    from_packed.encoded_data, from_views.encoded_data,
                    "{tag}: encoded bytes"
                );
                assert_eq!(
                    from_packed.header.null_count, from_views.header.null_count,
                    "{tag}: null count"
                );
                assert_eq!(
                    from_packed.header.cardinality, from_views.header.cardinality,
                    "{tag}: cardinality"
                );
                assert_eq!(
                    from_packed.header.min_value, from_views.header.min_value,
                    "{tag}: minimum"
                );
                assert_eq!(
                    from_packed.header.max_value, from_views.header.max_value,
                    "{tag}: maximum"
                );
                assert_eq!(
                    from_packed.header.is_sorted, from_views.header.is_sorted,
                    "{tag}: sorted flag"
                );
                assert_eq!(from_packed.min_row, from_views.min_row, "{tag}: min row");
                assert_eq!(from_packed.max_row, from_views.max_row, "{tag}: max row");
                assert_eq!(from_packed.ndv, from_views.ndv, "{tag}: distinct estimate");
                assert_eq!(
                    from_packed.null_bitmap, from_views.null_bitmap,
                    "{tag}: null bitmap"
                );
                assert_eq!(
                    from_packed.zone_maps.len(),
                    from_views.zone_maps.len(),
                    "{tag}: zone count"
                );
                assert_eq!(
                    from_packed.bloom_filter.as_ref().map(|b| b.to_bytes()),
                    from_views.bloom_filter.as_ref().map(|b| b.to_bytes()),
                    "{tag}: bloom"
                );
            }
        }
    }

    /// A packed build refuses input it cannot read rather than reading past
    /// the buffer or inventing rows
    #[test]
    fn a_packed_build_refuses_a_buffer_that_does_not_hold_its_rows() {
        let values = vec![0u8; 8 * 4];
        let options = SegmentOptions::default();
        assert!(
            ColumnSegment::build_packed(0, TypeId::Int64, 8, &values, &[], 5, options).is_err(),
            "five rows do not fit in four"
        );
        assert!(
            ColumnSegment::build_packed(0, TypeId::Int64, 0, &values, &[], 4, options).is_err(),
            "a packed build needs a width"
        );
        assert!(
            ColumnSegment::build_packed(0, TypeId::Int64, 8, &values, &[], 0, options).is_err(),
            "a segment needs rows"
        );
        assert!(
            ColumnSegment::build_packed(0, TypeId::Int64, 8, &values, &[], 4, options).is_ok(),
            "four rows fit in four"
        );
    }

    /// A bitmap longer than the rows it describes carries bits that belong
    /// to no row, and reading them as nulls would count rows the column
    /// does not have
    #[test]
    fn bits_past_the_last_row_are_not_nulls() {
        let rows = 5usize;
        let mut values = vec![0u8; rows * 8];
        for row in 0..rows {
            values[row * 8..(row + 1) * 8].copy_from_slice(&(row as i64).to_le_bytes());
        }
        // No row is null, but the three bits above the row count are set
        let segment = ColumnSegment::build_packed(
            0,
            TypeId::Int64,
            8,
            &values,
            &[0b1110_0000],
            rows,
            SegmentOptions::default(),
        )
        .expect("segment");
        assert_eq!(
            segment.header.null_count, 0,
            "a bit above the last row is not a null"
        );
        assert!(
            segment.null_bitmap.is_empty(),
            "a column with no nulls carries no bitmap"
        );

        // And a bitmap that does name a row still counts only that row
        let mut nulled = values.clone();
        nulled[8..16].fill(0);
        let segment = ColumnSegment::build_packed(
            0,
            TypeId::Int64,
            8,
            &nulled,
            &[0b1110_0010],
            rows,
            SegmentOptions::default(),
        )
        .expect("segment");
        assert_eq!(segment.header.null_count, 1, "one row is null");
    }

    /// A float column's fold has to reproduce the total order the general
    /// comparator gives, including across zero and for the negatives that
    /// grow away from it
    #[test]
    fn folded_float_words_keep_the_total_order() {
        let f64s = [
            f64::NEG_INFINITY,
            -1.0e300,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            1.0e300,
            f64::INFINITY,
        ];
        for a in f64s {
            for b in f64s {
                let (ab, bb) = (a.to_le_bytes(), b.to_le_bytes());
                assert_eq!(
                    ordered_word(le_word(&ab), 8, SlotOrder::Ieee)
                        .cmp(&ordered_word(le_word(&bb), 8, SlotOrder::Ieee)),
                    compare_values_ordered(&ab, &bb, 8, SlotOrder::Ieee),
                    "f64 fold disagreed on {a} vs {b}"
                );
            }
        }
        let f32s = [
            f32::NEG_INFINITY,
            -1.0e30,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1.0e30,
            f32::INFINITY,
        ];
        for a in f32s {
            for b in f32s {
                let (ab, bb) = (a.to_le_bytes(), b.to_le_bytes());
                assert_eq!(
                    ordered_word(le_word(&ab), 4, SlotOrder::Ieee)
                        .cmp(&ordered_word(le_word(&bb), 4, SlotOrder::Ieee)),
                    compare_values_ordered(&ab, &bb, 4, SlotOrder::Ieee),
                    "f32 fold disagreed on {a} vs {b}"
                );
            }
        }
    }

    /// The two fixed-size types wider than one word keep the general
    /// comparator, and a segment built over them still bounds its zones
    #[test]
    fn cells_wider_than_one_word_still_bound_their_segment() {
        let width = 18;
        let values: Vec<Vec<u8>> = (0..3_000u32)
            .map(|row| {
                let mut cell = vec![0u8; width];
                cell[..4].copy_from_slice(&row.to_le_bytes());
                cell
            })
            .collect();
        let views: Vec<Option<&[u8]>> = values.iter().map(|v| Some(v.as_slice())).collect();
        let segment = ColumnSegment::build(0, TypeId::Inet, width, &views).expect("segment");
        assert_eq!(segment.min_row, Some(0), "the first row holds the minimum");
        assert_eq!(
            segment.max_row,
            Some(values.len() - 1),
            "the last row holds the maximum"
        );
        assert!(segment.header.is_sorted, "an ascending column reads sorted");
    }
    use crate::encoding::{Predicate, eval_predicate_on_raw, varlen_slice_rows};

    /// Reusing the trial encode's output is only sound if those bytes are
    /// the bytes a fresh encode would produce.
    ///
    /// The segment is built the way a lake write builds it, then the column
    /// is packed independently here and encoded through the encoder the
    /// header names. The two byte strings have to match exactly, across the
    /// shapes that reach every selection branch: a column the trial wins on,
    /// one it loses on and falls back to Unencoded, one that short-circuits
    /// to Dictionary before any trial, one that goes Constant, and columns
    /// carrying nulls so the zeroed placeholder slots are compared too
    #[test]
    fn reused_trial_output_is_identical_to_encoding_the_column_again() {
        fn check(label: &str, type_id: TypeId, values: Vec<Option<Vec<u8>>>) {
            let views: Vec<Option<&[u8]>> = values.iter().map(|v| v.as_deref()).collect();
            let value_size = type_id.fixed_size().unwrap_or(0);
            let segment = ColumnSegment::build_with_options(
                0,
                type_id,
                value_size,
                &views,
                SegmentOptions {
                    bloom: BloomPolicy::Auto,
                    exact_encoding: true,
                    distinct_sketch: false,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: build failed: {e}"));

            // The same buffer the fused pass packs: values at their slot,
            // null slots zeroed
            let mut raw = vec![0u8; views.len() * value_size];
            for (i, v) in views.iter().enumerate() {
                if let Some(v) = v {
                    raw[i * value_size..(i + 1) * value_size].copy_from_slice(v);
                }
            }
            let fresh = crate::encoding::create_encoding(segment.header.encoding_type)
                .encode(&raw, views.len(), value_size)
                .unwrap_or_else(|e| panic!("{label}: re-encode failed: {e}"));

            assert_eq!(
                segment.encoded_data, fresh,
                "{label}: the segment kept bytes that differ from encoding the column                  again through {:?}",
                segment.header.encoding_type
            );
            assert_eq!(
                segment.header.encoded_size as usize,
                segment.encoded_data.len(),
                "{label}: the header size has to describe the bytes actually kept"
            );
        }

        let rows = 4096usize;
        // Ascending and distinct, which FastLanes wins on
        check(
            "int64 ascending distinct",
            TypeId::Int64,
            (0..rows)
                .map(|i| Some((i as i64).to_le_bytes().to_vec()))
                .collect(),
        );
        // Pseudorandom and distinct, where the candidate can lose to raw
        check(
            "int64 scattered distinct",
            TypeId::Int64,
            (0..rows)
                .map(|i| {
                    let v = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    Some((v as i64).to_le_bytes().to_vec())
                })
                .collect(),
        );
        // Low cardinality, which short-circuits to Dictionary with no trial
        check(
            "int64 low cardinality",
            TypeId::Int64,
            (0..rows)
                .map(|i| Some(((i % 8) as i64).to_le_bytes().to_vec()))
                .collect(),
        );
        // One repeated value, which goes Constant
        check(
            "int64 constant",
            TypeId::Int64,
            (0..rows)
                .map(|_| Some(7i64.to_le_bytes().to_vec()))
                .collect(),
        );
        // Nulls scattered through, so the zeroed slots are covered
        check(
            "int64 distinct with nulls",
            TypeId::Int64,
            (0..rows)
                .map(|i| {
                    if i % 11 == 0 {
                        None
                    } else {
                        Some((i as i64).to_le_bytes().to_vec())
                    }
                })
                .collect(),
        );
        // Floats reach the ALP candidate rather than FastLanes
        check(
            "float64 ascending",
            TypeId::Float64,
            (0..rows)
                .map(|i| Some((i as f64 * 1.5).to_le_bytes().to_vec()))
                .collect(),
        );
        check(
            "int32 distinct",
            TypeId::Int32,
            (0..rows)
                .map(|i| Some((i as i32).to_le_bytes().to_vec()))
                .collect(),
        );
    }

    /// The rows a build reports as its extremes have to be the column's
    /// actual extremes under that column's own order, because a caller that
    /// needs the value at full width reads it back through them instead of
    /// comparing the column a second time.
    ///
    /// The bounds are folded out of the per-zone bounds, so the columns here
    /// are long enough to span several zones, and they are shuffled so the
    /// extremes do not sit in the first or last one. Every case is checked
    /// against a flat scan of the same values
    #[test]
    fn extreme_rows_are_the_columns_extremes_under_its_own_order() {
        fn check(label: &str, type_id: TypeId, values: Vec<Option<Vec<u8>>>) {
            let views: Vec<Option<&[u8]>> = values.iter().map(|v| v.as_deref()).collect();
            let width = type_id.fixed_size().unwrap_or(0);
            let order = slot_order(type_id);
            let segment = ColumnSegment::build(0, type_id, width, &views)
                .unwrap_or_else(|e| panic!("{label}: build failed: {e}"));

            let mut expect_min: Option<(usize, &[u8])> = None;
            let mut expect_max: Option<(usize, &[u8])> = None;
            for (row, value) in views.iter().enumerate() {
                let Some(value) = value else { continue };
                if expect_min
                    .is_none_or(|(_, cur)| compare_values_ordered(value, cur, width, order).is_lt())
                {
                    expect_min = Some((row, value));
                }
                if expect_max
                    .is_none_or(|(_, cur)| compare_values_ordered(value, cur, width, order).is_gt())
                {
                    expect_max = Some((row, value));
                }
            }

            match expect_min {
                Some((_, value)) => {
                    let row = segment
                        .min_row
                        .unwrap_or_else(|| panic!("{label}: no minimum row"));
                    assert_eq!(
                        views[row].expect("minimum row holds a value"),
                        value,
                        "{label}: the reported minimum is not the column's minimum"
                    );
                    assert_eq!(
                        segment.header.min_value,
                        value_to_stat_slot(value),
                        "{label}: the header slot has to describe the same value"
                    );
                }
                None => assert!(
                    segment.min_row.is_none(),
                    "{label}: an all null column has no minimum"
                ),
            }
            match expect_max {
                Some((_, value)) => {
                    let row = segment
                        .max_row
                        .unwrap_or_else(|| panic!("{label}: no maximum row"));
                    assert_eq!(
                        views[row].expect("maximum row holds a value"),
                        value,
                        "{label}: the reported maximum is not the column's maximum"
                    );
                    assert_eq!(
                        segment.header.max_value,
                        varlen_upper_slot(value),
                        "{label}: the header slot has to describe the same value"
                    );
                }
                None => assert!(
                    segment.max_row.is_none(),
                    "{label}: an all null column has no maximum"
                ),
            }

            // And every value the column holds sits inside the recorded
            // bounds, which is the question a pruner asks of them
            for value in views.iter().flatten() {
                assert!(
                    compare_value_to_slot(value, &segment.header.min_value, width, order).is_ge(),
                    "{label}: the minimum excludes a value the column holds"
                );
                assert!(
                    compare_value_to_slot(value, &segment.header.max_value, width, order).is_le(),
                    "{label}: the maximum excludes a value the column holds"
                );
            }
        }

        // Enough rows to span several zones, shuffled by a stride coprime
        // with the count so the extremes land mid-column
        let rows = 5_000usize;
        let scatter = |i: usize| (i * 2_731) % rows;

        check(
            "int64 straddling zero",
            TypeId::Int64,
            (0..rows)
                .map(|i| Some(((scatter(i) as i64) - 2_500).to_le_bytes().to_vec()))
                .collect(),
        );
        check(
            "uint64 above the sign bit",
            TypeId::UInt64,
            (0..rows)
                .map(|i| Some((u64::MAX - scatter(i) as u64).to_le_bytes().to_vec()))
                .collect(),
        );
        check(
            "float64 straddling zero",
            TypeId::Float64,
            (0..rows)
                .map(|i| Some(((scatter(i) as f64) - 2_500.5).to_le_bytes().to_vec()))
                .collect(),
        );
        check(
            "timestamp before and after the epoch",
            TypeId::Timestamp,
            (0..rows)
                .map(|i| {
                    Some(
                        ((scatter(i) as i64 - 2_500) * 86_400_000_000)
                            .to_le_bytes()
                            .to_vec(),
                    )
                })
                .collect(),
        );
        // Values agreeing over the whole 32-byte slot, where the order is
        // decided past the prefix a slot can hold
        check(
            "varchar sharing a slot-length prefix",
            TypeId::Varchar,
            (0..rows)
                .map(|i| {
                    let mut value = vec![b'p'; STAT_VALUE_SIZE];
                    value.extend_from_slice(format!("{:08}", scatter(i)).as_bytes());
                    Some(value)
                })
                .collect(),
        );
        check(
            "varchar of mixed lengths",
            TypeId::Varchar,
            (0..rows)
                .map(|i| Some(format!("{}", scatter(i)).into_bytes()))
                .collect(),
        );
        // A column with nulls scattered through, including a whole zone of
        // them so a zone closes with no value in it
        check(
            "int32 with a fully null zone",
            TypeId::Int32,
            (0..rows)
                .map(|i| {
                    if (1_024..2_048).contains(&i) || i % 7 == 0 {
                        None
                    } else {
                        Some((scatter(i) as i32 - 2_500).to_le_bytes().to_vec())
                    }
                })
                .collect(),
        );
        check(
            "int64 all null",
            TypeId::Int64,
            (0..rows).map(|_| None).collect(),
        );
        check(
            "varchar all null",
            TypeId::Varchar,
            (0..rows).map(|_| None).collect(),
        );
    }

    /// The sketch is the only whole-column distinct count, and it is built
    /// only when the caller asks. The header's own cardinality answers a
    /// different question and still stops at the bloom threshold
    #[test]
    fn the_distinct_estimate_is_built_only_when_the_caller_asks_for_it() {
        let rows = 20_000usize;
        let owned: Vec<Vec<u8>> = (0..rows)
            .map(|i| (i as i64).to_le_bytes().to_vec())
            .collect();
        let views: Vec<Option<&[u8]>> = owned.iter().map(|v| Some(v.as_slice())).collect();

        let without = ColumnSegment::build_with_options(
            0,
            TypeId::Int64,
            8,
            &views,
            SegmentOptions {
                bloom: BloomPolicy::Auto,
                exact_encoding: false,
                distinct_sketch: false,
            },
        )
        .expect("build");
        assert!(
            without.ndv.is_none(),
            "a build that was not asked for an estimate must not carry one"
        );
        assert_eq!(
            without.header.cardinality,
            BLOOM_MIN_CARDINALITY + 1,
            "the exact count stops one past the bloom threshold"
        );

        let with = ColumnSegment::build_with_options(
            0,
            TypeId::Int64,
            8,
            &views,
            SegmentOptions {
                bloom: BloomPolicy::Auto,
                exact_encoding: false,
                distinct_sketch: true,
            },
        )
        .expect("build");
        let ndv = with.ndv.expect("an estimate was asked for");
        let error = (ndv as f64 - rows as f64).abs() / rows as f64;
        assert!(
            error < 0.05,
            "estimate {ndv} for {rows} distinct values is off by {error}"
        );
        assert_eq!(
            with.header.cardinality, without.header.cardinality,
            "asking for the estimate must not change the capped count"
        );
        assert_eq!(
            with.encoded_data, without.encoded_data,
            "asking for the estimate must not change the bytes written"
        );

        // A column below the threshold is counted exactly either way
        let low: Vec<Vec<u8>> = (0..rows)
            .map(|i| ((i % 9) as i64).to_le_bytes().to_vec())
            .collect();
        let low_views: Vec<Option<&[u8]>> = low.iter().map(|v| Some(v.as_slice())).collect();
        for sketch in [false, true] {
            let segment = ColumnSegment::build_with_options(
                0,
                TypeId::Int64,
                8,
                &low_views,
                SegmentOptions {
                    bloom: BloomPolicy::Auto,
                    exact_encoding: false,
                    distinct_sketch: sketch,
                },
            )
            .expect("build");
            assert_eq!(
                segment.header.cardinality, 9,
                "a column under the threshold is counted exactly"
            );
            if sketch {
                assert_eq!(segment.ndv, Some(9), "and the sketch agrees down here");
            }
        }
    }

    // -- Variable-length segment tests --

    #[test]
    fn test_varlen_segment_unencoded_roundtrip() {
        // Distinct strings of varying length plus a null and an empty string.
        let raw: Vec<Option<&[u8]>> = vec![
            Some(b"alpha".as_slice()),
            Some(b"".as_slice()),
            None,
            Some(b"a-much-longer-string-value-here".as_slice()),
            Some(b"beta".as_slice()),
        ];
        let seg = ColumnSegment::build(7, TypeId::Text, 0, &raw).expect("build varlen");
        assert_eq!(seg.header.null_count, 1);
        // Bit 2 is the null row; empty string at row 1 is not null.
        assert_eq!(seg.null_bitmap[0] & 0b100, 0b100);
        assert_eq!(seg.null_bitmap[0] & 0b010, 0);

        let decoder = create_encoding(seg.header.encoding_type);
        let decoded = decoder
            .decode(&seg.encoded_data, raw.len(), 0)
            .expect("decode varlen");
        let rows = varlen_slice_rows(&decoded, raw.len()).expect("slice rows");
        assert_eq!(rows[0], b"alpha");
        assert_eq!(rows[1], b"");
        assert_eq!(rows[3], b"a-much-longer-string-value-here");
        assert_eq!(rows[4], b"beta");

        // Predicate pushdown on the canonical buffer.
        let mask = eval_predicate_on_raw(&decoded, raw.len(), 0, &Predicate::Equality(b"beta"))
            .expect("eval");
        assert_eq!(mask[0] & (1 << 4), 1 << 4);
        assert_eq!(mask[0] & 1, 0);
    }

    #[test]
    fn test_varlen_segment_constant_roundtrip() {
        let raw: Vec<Option<&[u8]>> = vec![Some(b"same".as_slice()); 2048];
        let seg = ColumnSegment::build(3, TypeId::Varchar, 0, &raw).expect("build varlen const");
        assert_eq!(seg.header.encoding_type, EncodingType::Constant);
        let decoder = create_encoding(seg.header.encoding_type);
        let decoded = decoder
            .decode(&seg.encoded_data, raw.len(), 0)
            .expect("decode const varlen");
        let rows = varlen_slice_rows(&decoded, raw.len()).expect("slice rows");
        assert_eq!(rows.len(), 2048);
        assert!(rows.iter().all(|r| *r == b"same"));
    }

    #[test]
    fn test_signed_segment_minmax_handles_negatives() {
        // i64 column with values straddling zero. The segment header min/max
        // must be the true signed extrema so a metadata MIN()/MAX() answered
        // from the header (decode_fixed_scalar reads them back signed) is
        // correct. An unsigned byte compare would rank -100 above +100.
        let vals: Vec<[u8; 8]> = [-100i64, -50, 0, 25, 100, -1, 99]
            .iter()
            .map(|v| v.to_le_bytes())
            .collect();
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        let seg = ColumnSegment::build(0, TypeId::Int64, 8, &values).expect("build");
        let min = i64::from_le_bytes(seg.header.min_value[..8].try_into().unwrap());
        let max = i64::from_le_bytes(seg.header.max_value[..8].try_into().unwrap());
        assert_eq!(min, -100, "signed min");
        assert_eq!(max, 100, "signed max");

        // i128 picosecond timestamp with a pre-1970 negative value: the
        // segment min must be the negative, not the largest positive.
        let ps: Vec<[u8; 16]> = [
            -123_456_789_012_345i128,
            1_700_000_000_000_000_000_000i128,
            1_775_000_000_000_000_000_000i128,
            -1i128,
        ]
        .iter()
        .map(|v| v.to_le_bytes())
        .collect();
        let pv: Vec<Option<&[u8]>> = ps.iter().map(|v| Some(v.as_slice())).collect();
        // Logical type id is Timestamp (physical i128 ps), exactly what the
        // fold path passes to build.
        let pseg = ColumnSegment::build(1, TypeId::Timestamp, 16, &pv).expect("build ps");
        let pmin = i128::from_le_bytes(pseg.header.min_value[..16].try_into().unwrap());
        let pmax = i128::from_le_bytes(pseg.header.max_value[..16].try_into().unwrap());
        assert_eq!(pmin, -123_456_789_012_345i128, "ps signed min (pre-1970)");
        assert_eq!(pmax, 1_775_000_000_000_000_000_000i128, "ps signed max");

        // Unsigned columns keep unsigned ordering: a high-bit-set u64 is the
        // max, not a small positive.
        let uvals: Vec<[u8; 8]> = [1u64, 5, u64::MAX, 9, 0]
            .iter()
            .map(|v| v.to_le_bytes())
            .collect();
        let uv: Vec<Option<&[u8]>> = uvals.iter().map(|v| Some(v.as_slice())).collect();
        let useg = ColumnSegment::build(2, TypeId::UInt64, 8, &uv).expect("build u64");
        let umin = u64::from_le_bytes(useg.header.min_value[..8].try_into().unwrap());
        let umax = u64::from_le_bytes(useg.header.max_value[..8].try_into().unwrap());
        assert_eq!(umin, 0, "unsigned min");
        assert_eq!(umax, u64::MAX, "unsigned max stays unsigned");
    }

    #[test]
    fn test_varlen_segment_low_cardinality_picks_dictionary() {
        // Enum-like text column: a handful of distinct values over many rows
        // must select the variable-length dictionary, not FSST or Unencoded.
        let cats: [&[u8]; 3] = [b"active", b"inactive", b"suspended"];
        let raw: Vec<Option<&[u8]>> = (0..4096).map(|i| Some(cats[i % 3])).collect();
        let seg = ColumnSegment::build(9, TypeId::Text, 0, &raw).expect("build varlen dict");
        assert_eq!(seg.header.encoding_type, EncodingType::Dictionary);
        // Dictionary has an implicit lookup structure, so no bloom is built.
        assert!(seg.bloom_filter.is_none());

        let decoder = create_encoding(seg.header.encoding_type);
        let decoded = decoder
            .decode(&seg.encoded_data, raw.len(), 0)
            .expect("decode varlen dict");
        let rows = varlen_slice_rows(&decoded, raw.len()).expect("slice rows");
        for i in 0..raw.len() {
            assert_eq!(rows[i], cats[i % 3]);
        }
    }

    // -- SegmentHeader serialization tests --

    #[test]
    fn test_header_roundtrip_default() {
        let header = SegmentHeader {
            column_id: 0,
            encoding_type: EncodingType::Unencoded,
            raw_size: 0,
            encoded_size: 0,
            null_count: 0,
            cardinality: 0,
            min_value: [0u8; STAT_VALUE_SIZE],
            max_value: [0u8; STAT_VALUE_SIZE],
            data_checksum: 0,
            header_crc: 0,
            bloom_filter_offset: 0,
            bloom_filter_size: 0,
            is_sorted: false,
        };
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), SEGMENT_HEADER_SIZE);
        let recovered = SegmentHeader::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(recovered.column_id, 0);
        assert_eq!(recovered.encoding_type, EncodingType::Unencoded);
        assert_eq!(recovered.is_sorted, false);
    }

    #[test]
    fn test_header_roundtrip_populated() {
        let mut minVal = [0u8; STAT_VALUE_SIZE];
        minVal[STAT_VALUE_SIZE - 4..].copy_from_slice(&10u32.to_be_bytes());
        let mut maxVal = [0u8; STAT_VALUE_SIZE];
        maxVal[STAT_VALUE_SIZE - 4..].copy_from_slice(&999u32.to_be_bytes());

        let header = SegmentHeader {
            column_id: 42,
            encoding_type: EncodingType::FastLanes,
            raw_size: 81920,
            encoded_size: 40960,
            null_count: 7,
            cardinality: 500,
            min_value: minVal,
            max_value: maxVal,
            data_checksum: 8192,
            header_crc: 0,
            bloom_filter_offset: 49152,
            bloom_filter_size: 1024,
            is_sorted: true,
        };

        let bytes = header.to_bytes();
        let recovered = SegmentHeader::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(recovered.column_id, 42);
        assert_eq!(recovered.encoding_type, EncodingType::FastLanes);
        assert_eq!(recovered.raw_size, 81920);
        assert_eq!(recovered.encoded_size, 40960);
        assert_eq!(recovered.null_count, 7);
        assert_eq!(recovered.cardinality, 500);
        assert_eq!(recovered.min_value, minVal);
        assert_eq!(recovered.max_value, maxVal);
        assert_eq!(recovered.data_checksum, 8192);
        assert_eq!(recovered.bloom_filter_offset, 49152);
        assert_eq!(recovered.bloom_filter_size, 1024);
        assert_eq!(recovered.is_sorted, true);
    }

    #[test]
    fn test_header_roundtrip_all_encodings() {
        for encoding in 0..=7u8 {
            let encodingType = EncodingType::from_u8(encoding).expect("valid encoding");
            let header = SegmentHeader {
                column_id: encoding as u32,
                encoding_type: encodingType,
                raw_size: 0,
                encoded_size: 0,
                null_count: 0,
                cardinality: 0,
                min_value: [0u8; STAT_VALUE_SIZE],
                max_value: [0u8; STAT_VALUE_SIZE],
                data_checksum: 0,
                header_crc: 0,
                bloom_filter_offset: 0,
                bloom_filter_size: 0,
                is_sorted: false,
            };
            let bytes = header.to_bytes();
            let recovered = SegmentHeader::from_bytes(&bytes).expect("from_bytes failed");
            assert_eq!(recovered.encoding_type, encodingType);
            assert_eq!(recovered.column_id, encoding as u32);
        }
    }

    #[test]
    fn test_header_invalid_encoding_type() {
        let mut buf = [0u8; SEGMENT_HEADER_SIZE];
        buf[4] = 255; // invalid encoding type
        let result = SegmentHeader::from_bytes(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_reserved_bytes_zeroed() {
        let header = SegmentHeader {
            column_id: 1,
            encoding_type: EncodingType::Rle,
            raw_size: 100,
            encoded_size: 50,
            null_count: 0,
            cardinality: 10,
            min_value: [0u8; STAT_VALUE_SIZE],
            max_value: [0u8; STAT_VALUE_SIZE],
            data_checksum: 200,
            header_crc: 0,
            bloom_filter_offset: 0,
            bloom_filter_size: 0,
            is_sorted: false,
        };
        let bytes = header.to_bytes();

        // Reserved bytes [5..8] and [125..128] must be zero.
        assert_eq!(bytes[5], 0);
        assert_eq!(bytes[6], 0);
        assert_eq!(bytes[7], 0);
        assert_eq!(bytes[125], 0);
        assert_eq!(bytes[126], 0);
        assert_eq!(bytes[127], 0);
    }

    #[test]
    fn test_header_max_values() {
        let header = SegmentHeader {
            column_id: u32::MAX,
            encoding_type: EncodingType::Unencoded,
            raw_size: u64::MAX,
            encoded_size: u64::MAX,
            null_count: u64::MAX,
            cardinality: u64::MAX,
            min_value: [0xFF; STAT_VALUE_SIZE],
            max_value: [0xFF; STAT_VALUE_SIZE],
            data_checksum: u32::MAX,
            header_crc: 0,
            bloom_filter_offset: u64::MAX,
            bloom_filter_size: u32::MAX,
            is_sorted: true,
        };
        let bytes = header.to_bytes();
        let recovered = SegmentHeader::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(recovered.column_id, u32::MAX);
        assert_eq!(recovered.raw_size, u64::MAX);
        assert_eq!(recovered.encoded_size, u64::MAX);
        assert_eq!(recovered.null_count, u64::MAX);
        assert_eq!(recovered.cardinality, u64::MAX);
        assert_eq!(recovered.min_value, [0xFF; STAT_VALUE_SIZE]);
        assert_eq!(recovered.max_value, [0xFF; STAT_VALUE_SIZE]);
        assert_eq!(recovered.data_checksum, u32::MAX);
        assert_eq!(recovered.bloom_filter_offset, u64::MAX);
        assert_eq!(recovered.bloom_filter_size, u32::MAX);
        assert_eq!(recovered.is_sorted, true);
    }

    // -- ZoneMapEntry serialization tests --

    #[test]
    fn test_zone_map_roundtrip() {
        let mut minVal = [0u8; STAT_VALUE_SIZE];
        minVal[STAT_VALUE_SIZE - 2..].copy_from_slice(&50u16.to_be_bytes());
        let mut maxVal = [0u8; STAT_VALUE_SIZE];
        maxVal[STAT_VALUE_SIZE - 2..].copy_from_slice(&9999u16.to_be_bytes());

        let entry = ZoneMapEntry {
            min_value: minVal,
            max_value: maxVal,
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), ZONE_MAP_ENTRY_SIZE);

        let recovered = ZoneMapEntry::from_bytes(&bytes);
        assert_eq!(recovered.min_value, minVal);
        assert_eq!(recovered.max_value, maxVal);
    }

    #[test]
    fn test_zone_map_all_zeros() {
        let entry = ZoneMapEntry {
            min_value: [0u8; STAT_VALUE_SIZE],
            max_value: [0u8; STAT_VALUE_SIZE],
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes, [0u8; ZONE_MAP_ENTRY_SIZE]);
        let recovered = ZoneMapEntry::from_bytes(&bytes);
        assert_eq!(recovered.min_value, [0u8; STAT_VALUE_SIZE]);
        assert_eq!(recovered.max_value, [0u8; STAT_VALUE_SIZE]);
    }

    #[test]
    fn test_zone_map_all_ones() {
        let entry = ZoneMapEntry {
            min_value: [0xFF; STAT_VALUE_SIZE],
            max_value: [0xFF; STAT_VALUE_SIZE],
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes, [0xFF; ZONE_MAP_ENTRY_SIZE]);
        let recovered = ZoneMapEntry::from_bytes(&bytes);
        assert_eq!(recovered.min_value, [0xFF; STAT_VALUE_SIZE]);
        assert_eq!(recovered.max_value, [0xFF; STAT_VALUE_SIZE]);
    }

    // -- value_to_stat_slot tests --

    #[test]
    fn test_value_to_stat_slot_small_value() {
        let val = [1u8, 2, 3, 4];
        let slot = value_to_stat_slot(&val);
        // Value at the start, trailing bytes are zero padding.
        assert_eq!(&slot[..4], &[1, 2, 3, 4]);
        for i in 4..STAT_VALUE_SIZE {
            assert_eq!(slot[i], 0);
        }
    }

    #[test]
    fn test_value_to_stat_slot_exact_size() {
        let val = [0xAB; STAT_VALUE_SIZE];
        let slot = value_to_stat_slot(&val);
        assert_eq!(slot, [0xAB; STAT_VALUE_SIZE]);
    }

    #[test]
    fn test_value_to_stat_slot_oversized_value() {
        // Values larger than STAT_VALUE_SIZE are truncated to the first STAT_VALUE_SIZE bytes.
        let val = [0xFF; STAT_VALUE_SIZE + 10];
        let slot = value_to_stat_slot(&val);
        assert_eq!(slot, [0xFF; STAT_VALUE_SIZE]);
    }

    #[test]
    fn test_value_to_stat_slot_empty() {
        let slot = value_to_stat_slot(&[]);
        assert_eq!(slot, [0u8; STAT_VALUE_SIZE]);
    }

    // -- ColumnSegment::build tests --

    #[test]
    fn test_build_segment_basic() {
        let vals: Vec<[u8; 4]> = (0..100u32).map(|v| v.to_le_bytes()).collect();
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();

        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.header.column_id, 0);
        assert_eq!(segment.header.null_count, 0);
        // Bounded distinct tracking saturates at BLOOM_MIN_CARDINALITY+1 for
        // high-cardinality columns, downstream consumers read the saturated
        // value as "at least this many distinct values"
        assert!(segment.header.cardinality >= BLOOM_MIN_CARDINALITY);
        assert_eq!(segment.header.raw_size, 400);
        assert!(segment.header.encoded_size > 0);
        assert!(segment.header.is_sorted);
        assert!(segment.null_bitmap.is_empty());
        // Bloom filter is auto-built when cardinality >= BLOOM_MIN_CARDINALITY (64).
        // 100 distinct values exceeds the threshold.
        assert!(segment.bloom_filter.is_some());
        assert!(!segment.zone_maps.is_empty());
    }

    #[test]
    fn test_build_segment_with_nulls() {
        let vals: Vec<[u8; 4]> = (0..50u32).map(|v| v.to_le_bytes()).collect();
        let mut values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        // Insert 10 nulls at the end.
        for _ in 0..10 {
            values.push(None);
        }

        let segment = ColumnSegment::build(1, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.header.null_count, 10);
        assert_eq!(segment.header.cardinality, 50);
        assert!(!segment.null_bitmap.is_empty());

        // Verify null bitmap: first 50 rows are non-null (bits clear),
        // rows 50..59 are null (bits set).
        for i in 0..50 {
            let byteIdx = i / 8;
            let bitIdx = i % 8;
            assert_eq!(
                segment.null_bitmap[byteIdx] & (1 << bitIdx),
                0,
                "row {} should not be null",
                i
            );
        }
        for i in 50..60 {
            let byteIdx = i / 8;
            let bitIdx = i % 8;
            assert_ne!(
                segment.null_bitmap[byteIdx] & (1 << bitIdx),
                0,
                "row {} should be null",
                i
            );
        }
    }

    #[test]
    fn test_build_segment_all_nulls() {
        let values: Vec<Option<&[u8]>> = vec![None; 100];
        let segment = ColumnSegment::build(2, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.header.null_count, 100);
        assert_eq!(segment.header.cardinality, 0);
        assert_eq!(segment.header.min_value, [0u8; STAT_VALUE_SIZE]);
        assert_eq!(segment.header.max_value, [0u8; STAT_VALUE_SIZE]);
    }

    #[test]
    fn test_build_segment_empty_fails() {
        let values: Vec<Option<&[u8]>> = Vec::new();
        let result = ColumnSegment::build(0, TypeId::Int32, 4, &values);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_segment_single_value() {
        let val = 42u32.to_le_bytes();
        let values: Vec<Option<&[u8]>> = vec![Some(&val)];
        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.header.cardinality, 1);
        assert_eq!(segment.header.null_count, 0);
        assert!(segment.header.is_sorted);
        assert_eq!(segment.zone_maps.len(), 1);
    }

    #[test]
    fn test_build_segment_unsorted_data() {
        let vals: Vec<[u8; 4]> = vec![
            100u32.to_le_bytes(),
            50u32.to_le_bytes(),
            200u32.to_le_bytes(),
            10u32.to_le_bytes(),
        ];
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();

        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert!(!segment.header.is_sorted);
    }

    #[test]
    fn test_build_segment_constant_value() {
        let val = 7u32.to_le_bytes();
        let values: Vec<Option<&[u8]>> = (0..500).map(|_| Some(val.as_slice())).collect();

        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.header.cardinality, 1);
        assert_eq!(segment.header.encoding_type, EncodingType::Constant);
        assert!(segment.header.is_sorted);
    }

    // -- Zone map construction tests --

    #[test]
    fn test_zone_map_count_exact_batch() {
        // Exactly one batch worth of rows produces one zone map.
        let batchSize = ZONE_MAP_BATCH_SIZE as usize;
        let vals: Vec<[u8; 4]> = (0..batchSize as u32).map(|v| v.to_le_bytes()).collect();
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();

        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.zone_maps.len(), 1);
    }

    #[test]
    fn test_zone_map_count_partial_batch() {
        // One more row than ZONE_MAP_BATCH_SIZE produces two zone maps.
        let batchSize = ZONE_MAP_BATCH_SIZE as usize;
        let totalRows = batchSize + 1;
        let vals: Vec<[u8; 4]> = (0..totalRows as u32).map(|v| v.to_le_bytes()).collect();
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();

        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.zone_maps.len(), 2);
    }

    #[test]
    fn test_zone_map_min_max_values() {
        // Two batches with known u32 ranges that do not overlap.
        // Batch 0: values 0..1024, Batch 1: values 5000..6024.
        // Zone map comparison uses stat slots (right-padded LE bytes).
        let batchSize = ZONE_MAP_BATCH_SIZE as usize;
        let totalRows = batchSize * 2;

        let vals: Vec<[u8; 4]> = (0..totalRows)
            .map(|i| {
                let v: u32 = if i < batchSize {
                    i as u32
                } else {
                    5000 + (i - batchSize) as u32
                };
                v.to_le_bytes()
            })
            .collect();
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();

        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.zone_maps.len(), 2);

        // Batch 0 min = 0, Batch 1 min = 5000. Stat slots differ.
        let expectedMin0 = value_to_stat_slot(&0u32.to_le_bytes());
        let expectedMax0 = value_to_stat_slot(&1023u32.to_le_bytes());
        let expectedMin1 = value_to_stat_slot(&5000u32.to_le_bytes());

        assert_eq!(segment.zone_maps[0].min_value, expectedMin0);
        assert_eq!(segment.zone_maps[0].max_value, expectedMax0);
        assert_eq!(segment.zone_maps[1].min_value, expectedMin1);
    }

    #[test]
    fn test_zone_map_with_nulls_in_batch() {
        // A batch where some rows are null. Zone map should only reflect non-null values.
        let vals: Vec<[u8; 4]> = vec![
            10u32.to_le_bytes(),
            20u32.to_le_bytes(),
            30u32.to_le_bytes(),
        ];
        let values: Vec<Option<&[u8]>> =
            vec![Some(&vals[0]), None, Some(&vals[1]), None, Some(&vals[2])];

        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.zone_maps.len(), 1);
        let expectedMin = value_to_stat_slot(&10u32.to_le_bytes());
        let expectedMax = value_to_stat_slot(&30u32.to_le_bytes());
        assert_eq!(segment.zone_maps[0].min_value, expectedMin);
        assert_eq!(segment.zone_maps[0].max_value, expectedMax);
    }

    #[test]
    fn test_zone_map_all_null_batch() {
        // All-null zones use sentinel min=0xFF/max=0x00 so range queries skip them.
        let values: Vec<Option<&[u8]>> = vec![None; 10];
        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        assert_eq!(segment.zone_maps.len(), 1);
        assert_eq!(segment.zone_maps[0].min_value, [0xFF; STAT_VALUE_SIZE]);
        assert_eq!(segment.zone_maps[0].max_value, [0u8; STAT_VALUE_SIZE]);
    }

    // -- Header + segment integration tests --

    #[test]
    fn test_build_then_serialize_header() {
        // Use single-byte values 0..200 to test sorted detection.
        // Single-byte LE values have consistent raw byte ordering.
        let vals: Vec<[u8; 1]> = (0..200u8).map(|v| [v]).collect();
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();

        let segment = ColumnSegment::build(5, TypeId::Int32, 1, &values).expect("build failed");

        let headerBytes = segment.header.to_bytes();
        let recovered = SegmentHeader::from_bytes(&headerBytes).expect("from_bytes failed");

        assert_eq!(recovered.column_id, 5);
        assert_eq!(recovered.null_count, 0);
        // Bounded distinct tracking saturates at BLOOM_MIN_CARDINALITY+1 for
        // high-cardinality columns
        assert!(recovered.cardinality >= BLOOM_MIN_CARDINALITY);
        assert_eq!(recovered.raw_size, 200);
        assert!(recovered.is_sorted);
    }

    #[test]
    fn test_zone_map_serialization_roundtrip() {
        let vals: Vec<[u8; 4]> = (0..2048u32).map(|v| v.to_le_bytes()).collect();
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();

        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        for entry in &segment.zone_maps {
            let bytes = entry.to_bytes();
            let recovered = ZoneMapEntry::from_bytes(&bytes);
            assert_eq!(recovered.min_value, entry.min_value);
            assert_eq!(recovered.max_value, entry.max_value);
        }
    }

    #[test]
    fn test_segment_data_offset_initially_zero() {
        let val = 1u32.to_le_bytes();
        let values: Vec<Option<&[u8]>> = vec![Some(&val); 100];
        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");

        // build() stamps the payload checksum; bloom_filter_offset is set by
        // the file writer, not build().
        assert_eq!(
            segment.header.data_checksum,
            zyron_common::hash32(&segment.encoded_data)
        );
        assert_eq!(segment.header.bloom_filter_offset, 0);
        assert_eq!(segment.header.bloom_filter_size, 0);
    }
}
