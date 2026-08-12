//! Column segment: the atomic unit of columnar storage.
//!
//! A ColumnSegment holds one column's data for a contiguous range of rows,
//! including encoding metadata, zone maps for segment pruning, and an
//! optional bloom filter for point lookups.

use crate::columnar::bloom::BloomFilter;
use crate::columnar::constants::*;
use crate::encoding::{
    EncodingType, create_encoding, select_encoding, select_encoding_exact, select_encoding_varlen,
    varlen_pack,
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

        // Single fused pass over values, computes:
        //   - null count and null bitmap
        //   - global min/max stat slots
        //   - sorted flag
        //   - distinct tracking, capped at BLOOM_MIN_CARDINALITY+1 since we
        //     only need to know whether the column is high-cardinality enough
        //     to warrant a bloom filter, exact count beyond the threshold is
        //     not useful for query planning at the segment level
        //   - raw data buffer with non-null values copied into their slots,
        //     buffer is allocated with zero-fill so null slots stay zeroed
        //     for encoder determinism
        //   - per-zone min/max emitted at every ZONE_MAP_BATCH_SIZE boundary
        //
        // Prior implementation walked values 4-5 separate times which kept
        // the L1/L2 cache cold for each pass on large columns, the fused
        // pass touches each value once and keeps zone-map and bitmap state
        // in registers
        let rawSize = (rowCount * valueSize) as u64;
        // SAFETY: the fused pass below writes every one of `buf_len` slots
        // (null slots explicitly zeroed, non-null slots copied) before the
        // encoder reads rawData; no path reads it before the fill. Zeroing
        // up front would memset the whole column buffer only to overwrite
        // it, regressing encode/compaction throughput on large columns.
        let buf_len = rowCount * valueSize;
        #[allow(clippy::uninit_vec)]
        let mut rawData: Vec<u8> = {
            let mut v = Vec::with_capacity(buf_len);
            unsafe { v.set_len(buf_len) };
            v
        };

        // Two's complement ordering for signed columns so a negative value
        // (incl. a pre-1970 picosecond timestamp) sorts below zero in the
        // segment min/max instead of above every positive under an unsigned
        // byte compare.
        let statOrder = slot_order(typeId);

        let batchSize = ZONE_MAP_BATCH_SIZE as usize;
        let zoneCount = rowCount.div_ceil(batchSize);
        let mut zoneMaps: Vec<ZoneMapEntry> = Vec::with_capacity(zoneCount);

        let mut nullCount = 0u64;
        let mut nullBitmap: Vec<u8> = Vec::new();
        // Bounded distinct tracking, capped at BLOOM_MIN_CARDINALITY+1 since
        // we only need to distinguish "high enough cardinality for bloom" from
        // "low cardinality". For high-cardinality columns this saves an
        // unbounded HashSet that would otherwise grow to N entries (~32 MB
        // for 1M unique values), the segment header reports the saturated
        // value which downstream consumers treat as "at least this many"
        let mut distinct = hashbrown::HashSet::new();
        let mut distinctCapped = false;
        let mut globalMin: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut globalMax: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut isSorted = true;
        let mut prevRaw: Option<&[u8]> = None;

        let mut zoneMin: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut zoneMax: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut zoneIdx = 0usize;

        for (i, val) in values.iter().enumerate() {
            let nextZoneBoundary = (zoneIdx + 1) * batchSize;
            if i == nextZoneBoundary {
                zoneMaps.push(ZoneMapEntry {
                    min_value: zoneMin.unwrap_or([0xFF; STAT_VALUE_SIZE]),
                    max_value: zoneMax.unwrap_or([0u8; STAT_VALUE_SIZE]),
                });
                zoneMin = None;
                zoneMax = None;
                zoneIdx += 1;
            }

            match val {
                None => {
                    nullCount += 1;
                    if nullBitmap.is_empty() {
                        nullBitmap = vec![0u8; rowCount.div_ceil(8)];
                    }
                    nullBitmap[i / 8] |= 1 << (i % 8);
                    // Zero the null slot in rawData since the buffer was
                    // allocated uninitialized, encoders treat the slot as
                    // a deterministic zero placeholder and the null bitmap
                    // tells consumers to skip it
                    let start = i * valueSize;
                    let end = start + valueSize;
                    rawData[start..end].fill(0);
                }
                Some(v) => {
                    if !distinctCapped {
                        distinct.insert(*v);
                        if distinct.len() as u64 > BLOOM_MIN_CARDINALITY {
                            distinctCapped = true;
                        }
                    }
                    // Compare the raw value against each running bound and
                    // build the 32-byte slot only on a row that actually moves
                    // a bound, instead of materializing a slot for every row.
                    use std::cmp::Ordering::{Equal, Greater, Less};
                    // A variable-length value longer than a slot is held
                    // as a prefix, and its upper bound is that prefix
                    // rounded up, so it can exceed a recorded maximum its
                    // prefix merely ties with
                    let truncated = valueSize == 0 && v.len() > STAT_VALUE_SIZE;
                    let below = |cur: &[u8; STAT_VALUE_SIZE]| {
                        compare_value_to_slot(v, cur, valueSize, statOrder) == Less
                    };
                    let above = |cur: &[u8; STAT_VALUE_SIZE]| {
                        match compare_value_to_slot(v, cur, valueSize, statOrder) {
                            Greater => true,
                            Equal => truncated,
                            Less => false,
                        }
                    };
                    let new_gmin = globalMin.is_none_or(|cur| below(&cur));
                    let new_gmax = globalMax.is_none_or(|cur| above(&cur));
                    let new_zmin = zoneMin.is_none_or(|cur| below(&cur));
                    let new_zmax = zoneMax.is_none_or(|cur| above(&cur));
                    if new_gmin || new_gmax || new_zmin || new_zmax {
                        let lower = value_to_stat_slot(v);
                        let upper = if truncated {
                            varlen_upper_slot(v)
                        } else {
                            lower
                        };
                        if new_gmin {
                            globalMin = Some(lower);
                        }
                        if new_gmax {
                            globalMax = Some(upper);
                        }
                        if new_zmin {
                            zoneMin = Some(lower);
                        }
                        if new_zmax {
                            zoneMax = Some(upper);
                        }
                    }

                    // A variable-length column orders by its bytes from the
                    // first, and compare_le_bytes reads fixed-width values
                    // from the last and requires equal lengths
                    let descends = if valueSize == 0 {
                        prevRaw.is_some_and(|prev| *v < prev)
                    } else {
                        prevRaw.is_some_and(|prev| {
                            compare_le_bytes(v, prev) == std::cmp::Ordering::Less
                        })
                    };
                    if isSorted && descends {
                        isSorted = false;
                    }
                    prevRaw = Some(*v);

                    let start = i * valueSize;
                    let end = start + valueSize;
                    let raw_len = rawData.len();
                    if v.len() != valueSize || end > raw_len {
                        // A non-null value whose width does not match the fixed
                        // column value size cannot be packed into its slot, fail
                        // instead of zero-filling and corrupting the value
                        return Err(ZyronError::EncodingFailed(format!(
                            "non-null value at row {} has length {} expected {}",
                            i,
                            v.len(),
                            valueSize
                        )));
                    }
                    rawData[start..end].copy_from_slice(v);
                }
            }
        }
        // Push the final zone, may be partially filled
        if zoneMaps.len() < zoneCount {
            zoneMaps.push(ZoneMapEntry {
                min_value: zoneMin.unwrap_or([0xFF; STAT_VALUE_SIZE]),
                max_value: zoneMax.unwrap_or([0u8; STAT_VALUE_SIZE]),
            });
        }

        let cardinality = distinct.len() as u64;
        let minValue = globalMin.unwrap_or([0u8; STAT_VALUE_SIZE]);
        let maxValue = globalMax.unwrap_or([0u8; STAT_VALUE_SIZE]);

        let encodingType = if options.exact_encoding {
            select_encoding_exact(typeId, values)
        } else {
            select_encoding(typeId, values)
        };
        let encoder = create_encoding(encodingType);

        let encodedData = encoder.encode(&rawData, rowCount, valueSize)?;
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
        })
    }

    /// Builds a variable-length column segment. Values are stored in the
    /// canonical variable-length buffer (a u32 offset array plus a values
    /// blob). Zone-map and segment min/max use a left-aligned, zero-padded
    /// byte prefix into STAT_VALUE_SIZE compared lexicographically. The same
    /// prefix transform is applied to predicate literals at prune time, so a
    /// shared prefix only ever widens a zone (a conservative non-skip), never
    /// a false skip. The null bitmap is authoritative, so an empty string and
    /// a null stay distinct.
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
        let mut distinct = hashbrown::HashSet::new();
        let mut distinctCapped = false;
        let mut globalMin: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut globalMax: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut isSorted = true;
        let mut prevRaw: Option<&[u8]> = None;

        let mut zoneMin: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut zoneMax: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut zoneIdx = 0usize;

        for (i, val) in values.iter().enumerate() {
            let nextZoneBoundary = (zoneIdx + 1) * batchSize;
            if i == nextZoneBoundary {
                zoneMaps.push(ZoneMapEntry {
                    min_value: zoneMin.unwrap_or([0xFF; STAT_VALUE_SIZE]),
                    max_value: zoneMax.unwrap_or([0u8; STAT_VALUE_SIZE]),
                });
                zoneMin = None;
                zoneMax = None;
                zoneIdx += 1;
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
                    if !distinctCapped {
                        distinct.insert(*v);
                        if distinct.len() as u64 > BLOOM_MIN_CARDINALITY {
                            distinctCapped = true;
                        }
                    }
                    // Variable-length min/max order the stat slots
                    // lexicographically (prefix bound, correct for string and
                    // binary). Compare the value's padded-prefix order against
                    // each running bound and only build the 32-byte slot on a
                    // row that actually moves a bound.
                    use std::cmp::Ordering::{Greater, Less};
                    let lex = |v: &[u8], cur: &[u8; STAT_VALUE_SIZE]| -> std::cmp::Ordering {
                        for i in 0..STAT_VALUE_SIZE {
                            match v.get(i).copied().unwrap_or(0).cmp(&cur[i]) {
                                std::cmp::Ordering::Equal => continue,
                                other => return other,
                            }
                        }
                        std::cmp::Ordering::Equal
                    };
                    let new_gmin = globalMin.is_none_or(|cur| lex(v, &cur) == Less);
                    let new_gmax = globalMax.is_none_or(|cur| lex(v, &cur) == Greater);
                    let new_zmin = zoneMin.is_none_or(|cur| lex(v, &cur) == Less);
                    let new_zmax = zoneMax.is_none_or(|cur| lex(v, &cur) == Greater);
                    if new_gmin || new_gmax || new_zmin || new_zmax {
                        let slot = value_to_stat_slot(v);
                        if new_gmin {
                            globalMin = Some(slot);
                        }
                        if new_gmax {
                            globalMax = Some(slot);
                        }
                        if new_zmin {
                            zoneMin = Some(slot);
                        }
                        if new_zmax {
                            zoneMax = Some(slot);
                        }
                    }

                    if isSorted
                        && let Some(prev) = prevRaw
                        && (*v).cmp(prev) == std::cmp::Ordering::Less
                    {
                        isSorted = false;
                    }
                    prevRaw = Some(*v);
                }
            }
        }
        if zoneMaps.len() < zoneCount {
            zoneMaps.push(ZoneMapEntry {
                min_value: zoneMin.unwrap_or([0xFF; STAT_VALUE_SIZE]),
                max_value: zoneMax.unwrap_or([0u8; STAT_VALUE_SIZE]),
            });
        }

        let cardinality = distinct.len() as u64;
        let minValue = globalMin.unwrap_or([0u8; STAT_VALUE_SIZE]);
        let maxValue = globalMax.unwrap_or([0u8; STAT_VALUE_SIZE]);

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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{Predicate, eval_predicate_on_raw, varlen_slice_rows};

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
