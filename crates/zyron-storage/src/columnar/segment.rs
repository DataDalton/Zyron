//! Column segment: the atomic unit of columnar storage.
//!
//! A ColumnSegment holds one column's data for a contiguous range of rows,
//! including encoding metadata, zone maps for segment pruning, and an
//! optional bloom filter for point lookups.

use crate::columnar::bloom::BloomFilter;
use crate::columnar::constants::*;
use crate::encoding::{EncodingType, create_encoding, select_encoding};
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
    pub uncompressed_size: u64,
    /// Byte size of encoded column data.
    pub compressed_size: u64,
    /// Number of null values in this segment.
    pub null_count: u64,
    /// Number of distinct non-null values.
    pub cardinality: u64,
    /// Minimum value in the segment (left-padded with zeros).
    pub min_value: [u8; STAT_VALUE_SIZE],
    /// Maximum value in the segment (left-padded with zeros).
    pub max_value: [u8; STAT_VALUE_SIZE],
    /// Byte offset from start of file to encoded data.
    pub data_offset: u64,
    /// Byte offset from start of file to bloom filter (0 if absent).
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
        buf[8..16].copy_from_slice(&self.uncompressed_size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.compressed_size.to_le_bytes());
        buf[24..32].copy_from_slice(&self.null_count.to_le_bytes());
        buf[32..40].copy_from_slice(&self.cardinality.to_le_bytes());
        buf[40..72].copy_from_slice(&self.min_value);
        buf[72..104].copy_from_slice(&self.max_value);
        buf[104..112].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[112..120].copy_from_slice(&self.bloom_filter_offset.to_le_bytes());
        buf[120..124].copy_from_slice(&self.bloom_filter_size.to_le_bytes());
        buf[124] = if self.is_sorted { 1 } else { 0 };
        // [125..128] reserved

        buf
    }

    /// Deserializes a 128-byte little-endian buffer into a SegmentHeader.
    pub fn from_bytes(buf: &[u8; SEGMENT_HEADER_SIZE]) -> Result<Self> {
        let columnId = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let encodingType = EncodingType::from_u8(buf[4])?;
        let uncompressedSize = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);
        let compressedSize = u64::from_le_bytes([
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

        let dataOffset = u64::from_le_bytes([
            buf[104], buf[105], buf[106], buf[107], buf[108], buf[109], buf[110], buf[111],
        ]);
        let bloomFilterOffset = u64::from_le_bytes([
            buf[112], buf[113], buf[114], buf[115], buf[116], buf[117], buf[118], buf[119],
        ]);
        let bloomFilterSize = u32::from_le_bytes([buf[120], buf[121], buf[122], buf[123]]);
        let isSorted = buf[124] != 0;

        Ok(Self {
            column_id: columnId,
            encoding_type: encodingType,
            uncompressed_size: uncompressedSize,
            compressed_size: compressedSize,
            null_count: nullCount,
            cardinality,
            min_value: minValue,
            max_value: maxValue,
            data_offset: dataOffset,
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
    /// Builds a ColumnSegment from raw column values.
    ///
    /// `column_id` - ordinal position of this column in the table schema.
    /// `type_id` - data type of the column, used for encoding selection.
    /// `value_size` - byte width of each value (fixed-size types only).
    /// `values` - row values, where None represents a null.
    ///
    /// The bloom filter field is left as None. The caller attaches one
    /// separately if the cardinality warrants it.
    pub fn build(
        columnId: u32,
        typeId: TypeId,
        valueSize: usize,
        values: &[Option<&[u8]>],
    ) -> Result<Self> {
        let rowCount = values.len();
        if rowCount == 0 {
            return Err(ZyronError::EncodingFailed(
                "cannot build segment from zero rows".to_string(),
            ));
        }

        // Compute column statistics: min, max, null count, cardinality, sorted flag.
        // Sorted detection uses raw byte comparison (preserves LE numeric ordering).
        // Min/max use stat slots for fixed-size storage in the segment header.
        let mut nullCount = 0u64;
        let mut distinct = hashbrown::HashSet::new();
        let mut globalMin: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut globalMax: Option<[u8; STAT_VALUE_SIZE]> = None;
        let mut isSorted = true;
        let mut prevRaw: Option<&[u8]> = None;

        for val in values.iter() {
            match val {
                None => nullCount += 1,
                Some(v) => {
                    distinct.insert(*v);
                    let slot = value_to_stat_slot(v);

                    globalMin = Some(match globalMin {
                        Some(cur)
                            if compare_stat_slots(&cur, &slot) != std::cmp::Ordering::Greater =>
                        {
                            cur
                        }
                        _ => slot,
                    });
                    globalMax = Some(match globalMax {
                        Some(cur)
                            if compare_stat_slots(&cur, &slot) != std::cmp::Ordering::Less =>
                        {
                            cur
                        }
                        _ => slot,
                    });

                    if isSorted
                        && let Some(prev) = prevRaw
                        && compare_le_bytes(v, prev) == std::cmp::Ordering::Less
                    {
                        isSorted = false;
                    }
                    prevRaw = Some(*v);
                }
            }
        }

        let cardinality = distinct.len() as u64;
        let minValue = globalMin.unwrap_or([0u8; STAT_VALUE_SIZE]);
        let maxValue = globalMax.unwrap_or([0u8; STAT_VALUE_SIZE]);

        // Select encoding strategy based on type and sample statistics.
        let encodingType = select_encoding(typeId, values);
        let encoder = create_encoding(encodingType);

        // Build raw data buffer from non-null values, filling nulls with zeros.
        let uncompressedSize = (rowCount * valueSize) as u64;
        let mut rawData = vec![0u8; rowCount * valueSize];
        for (i, val) in values.iter().enumerate() {
            if let Some(v) = val {
                let start = i * valueSize;
                let end = start + valueSize;
                if v.len() == valueSize && end <= rawData.len() {
                    rawData[start..end].copy_from_slice(v);
                }
            }
        }

        let encodedData = encoder.encode(&rawData, rowCount, valueSize)?;
        let compressedSize = encodedData.len() as u64;

        // Build null bitmap if any nulls exist. Bit i is set when row i is null.
        let nullBitmap = if nullCount > 0 {
            let bitmapLen = rowCount.div_ceil(8);
            let mut bitmap = vec![0u8; bitmapLen];
            for (i, val) in values.iter().enumerate() {
                if val.is_none() {
                    bitmap[i / 8] |= 1 << (i % 8);
                }
            }
            bitmap
        } else {
            Vec::new()
        };

        // Build zone maps, one entry per ZONE_MAP_BATCH_SIZE rows.
        let batchSize = ZONE_MAP_BATCH_SIZE as usize;
        let zoneCount = rowCount.div_ceil(batchSize);
        let mut zoneMaps = Vec::with_capacity(zoneCount);

        for z in 0..zoneCount {
            let zoneStart = z * batchSize;
            let zoneEnd = (zoneStart + batchSize).min(rowCount);

            let mut zoneMin: Option<[u8; STAT_VALUE_SIZE]> = None;
            let mut zoneMax: Option<[u8; STAT_VALUE_SIZE]> = None;

            for v in values[zoneStart..zoneEnd].iter().flatten() {
                let slot = value_to_stat_slot(v);
                zoneMin = Some(match zoneMin {
                    Some(cur) if compare_stat_slots(&cur, &slot) != std::cmp::Ordering::Greater => {
                        cur
                    }
                    _ => slot,
                });
                zoneMax = Some(match zoneMax {
                    Some(cur) if compare_stat_slots(&cur, &slot) != std::cmp::Ordering::Less => cur,
                    _ => slot,
                });
            }

            // All-null zones use sentinel min=0xFF/max=0x00 (min > max is
            // impossible for real data, so range queries always skip them).
            zoneMaps.push(ZoneMapEntry {
                min_value: zoneMin.unwrap_or([0xFF; STAT_VALUE_SIZE]),
                max_value: zoneMax.unwrap_or([0u8; STAT_VALUE_SIZE]),
            });
        }

        // Build bloom filter when cardinality is high enough to benefit.
        // Dictionary-encoded segments already have an implicit lookup structure,
        // so bloom filters are skipped for those.
        let bloomFilter =
            if cardinality >= BLOOM_MIN_CARDINALITY && encodingType != EncodingType::Dictionary {
                let mut filter = BloomFilter::new(cardinality);
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
            uncompressed_size: uncompressedSize,
            compressed_size: compressedSize,
            null_count: nullCount,
            cardinality,
            min_value: minValue,
            max_value: maxValue,
            data_offset: 0,
            bloom_filter_offset: 0,
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

    // -- SegmentHeader serialization tests --

    #[test]
    fn test_header_roundtrip_default() {
        let header = SegmentHeader {
            column_id: 0,
            encoding_type: EncodingType::Unencoded,
            uncompressed_size: 0,
            compressed_size: 0,
            null_count: 0,
            cardinality: 0,
            min_value: [0u8; STAT_VALUE_SIZE],
            max_value: [0u8; STAT_VALUE_SIZE],
            data_offset: 0,
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
            uncompressed_size: 81920,
            compressed_size: 40960,
            null_count: 7,
            cardinality: 500,
            min_value: minVal,
            max_value: maxVal,
            data_offset: 8192,
            bloom_filter_offset: 49152,
            bloom_filter_size: 1024,
            is_sorted: true,
        };

        let bytes = header.to_bytes();
        let recovered = SegmentHeader::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(recovered.column_id, 42);
        assert_eq!(recovered.encoding_type, EncodingType::FastLanes);
        assert_eq!(recovered.uncompressed_size, 81920);
        assert_eq!(recovered.compressed_size, 40960);
        assert_eq!(recovered.null_count, 7);
        assert_eq!(recovered.cardinality, 500);
        assert_eq!(recovered.min_value, minVal);
        assert_eq!(recovered.max_value, maxVal);
        assert_eq!(recovered.data_offset, 8192);
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
                uncompressed_size: 0,
                compressed_size: 0,
                null_count: 0,
                cardinality: 0,
                min_value: [0u8; STAT_VALUE_SIZE],
                max_value: [0u8; STAT_VALUE_SIZE],
                data_offset: 0,
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
            uncompressed_size: 100,
            compressed_size: 50,
            null_count: 0,
            cardinality: 10,
            min_value: [0u8; STAT_VALUE_SIZE],
            max_value: [0u8; STAT_VALUE_SIZE],
            data_offset: 200,
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
            uncompressed_size: u64::MAX,
            compressed_size: u64::MAX,
            null_count: u64::MAX,
            cardinality: u64::MAX,
            min_value: [0xFF; STAT_VALUE_SIZE],
            max_value: [0xFF; STAT_VALUE_SIZE],
            data_offset: u64::MAX,
            bloom_filter_offset: u64::MAX,
            bloom_filter_size: u32::MAX,
            is_sorted: true,
        };
        let bytes = header.to_bytes();
        let recovered = SegmentHeader::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(recovered.column_id, u32::MAX);
        assert_eq!(recovered.uncompressed_size, u64::MAX);
        assert_eq!(recovered.compressed_size, u64::MAX);
        assert_eq!(recovered.null_count, u64::MAX);
        assert_eq!(recovered.cardinality, u64::MAX);
        assert_eq!(recovered.min_value, [0xFF; STAT_VALUE_SIZE]);
        assert_eq!(recovered.max_value, [0xFF; STAT_VALUE_SIZE]);
        assert_eq!(recovered.data_offset, u64::MAX);
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
        assert_eq!(segment.header.cardinality, 100);
        assert_eq!(segment.header.uncompressed_size, 400);
        assert!(segment.header.compressed_size > 0);
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
        assert_eq!(recovered.cardinality, 200);
        assert_eq!(recovered.uncompressed_size, 200);
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

        // data_offset and bloom_filter_offset are set by the file writer, not build().
        assert_eq!(segment.header.data_offset, 0);
        assert_eq!(segment.header.bloom_filter_offset, 0);
        assert_eq!(segment.header.bloom_filter_size, 0);
    }
}
