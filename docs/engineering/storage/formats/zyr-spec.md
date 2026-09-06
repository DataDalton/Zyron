# .zyr Columnar File Format Specification

Format version 1.1. The running binary writes 1.1 and reads 1.0 and 1.1.

## Overview

A `.zyr` file holds one table's rows in columnar form: one encoded segment per column, a zone map and an optional bloom filter per segment, and a segment index at the end of the file. Compaction writes these files from heap rows, and the lake reader opens them with three reads, one for the header, one for the index and trailer, and then only the parts of the segments a query touches.

The file carries the 20-byte envelope header every persistent Zyron file starts with, and owns its own trailer instead of an envelope footer checksum. The registry reports the kind `ZyrColumnar` with framing `own_trailer`.

All multi-byte integers are little-endian.

## File layout

```text
[0, 512)              file header, 128 bytes of fields then zero padding
[512, index_offset)   column segments, each starting on a 64-byte boundary
[index_offset, ...)   segment index, 20 bytes per column
last 20 bytes         trailer: index offset, sentinel, index checksum
```

Nothing derives a segment position from the header size. Every segment offset comes from the segment index, which is what lets one reader serve both 1.0 and 1.1.

## File header

The header occupies 512 bytes (`FILE_HEADER_SIZE`). The first 128 bytes hold fields (`FILE_HEADER_METADATA_SIZE`), the remaining 384 bytes are zero-filled room for future fields.

### Envelope prefix, bytes 0 to 20

| Offset | Size | Type | Field | Value |
| ------ | ---- | ---- | ----- | ----- |
| 0 | 4 | bytes | magic | `ZCOL`, the magic allocated to `FormatKind::ZyrColumnar` |
| 4 | 2 | u16 | version_major | 1 |
| 6 | 2 | u16 | version_minor | 1 for a file written by the current binary, 0 for a 1.0 file |
| 8 | 4 | u32 | header_length | 128. The reader refuses any other value |
| 12 | 4 | u32 | flags | Always 0. The low byte is reserved to the substrate, the high three bytes to the format |
| 16 | 4 | u32 | header_checksum | Covers bytes 0 to 16 and 20 to 128, see Checksums |

### Format fields, bytes 20 to 128

| Offset | Size | Type | Field | Meaning |
| ------ | ---- | ---- | ----- | ------- |
| 20 | 4 | u32 | column_count | Segments the writer intended. The authoritative count is `segment_index_size / 20` |
| 24 | 8 | u64 | row_count | Rows in every segment of the file. Fixes the zone count and the null bitmap length |
| 32 | 8 | u64 | table_id | Owning table |
| 40 | 8 | u64 | xmin_range_lo | Lowest creating transaction id in the file |
| 48 | 8 | u64 | xmin_range_hi | Highest creating transaction id in the file |
| 56 | 8 | u64 | xmax_range_lo | Lowest superseding transaction id. The compaction writer records 0 |
| 64 | 8 | u64 | xmax_range_hi | Highest superseding transaction id. The compaction writer records 0 |
| 72 | 4 | u32 | primary_key_column_id | Column the file is clustered on |
| 76 | 1 | u8 | sort_order | 0 none, 1 ascending, 2 descending. Any other value is rejected |
| 77 | 3 | bytes | reserved | Zero |
| 80 | 8 | u64 | segment_index_offset | Absolute offset of the segment index. Written as 0 and rewritten when the file is finalized |
| 88 | 4 | u32 | segment_index_size | Bytes in the index region |
| 92 | 36 | bytes | reserved | Zero |
| 128 | 384 | bytes | padding | Zero, room to grow inside the 512-byte header |

## Column segments

Each segment starts on a 64-byte boundary (`SEGMENT_ALIGNMENT`). The segment header, the bloom blocks and the zone map entries are all cache-line multiples, so aligning the segment start keeps every part aligned. Nothing reads a segment as pages.

A segment is laid out in this order, and the writer zero-pads the end up to the next 64-byte boundary:

```text
segment header      128 bytes
bloom filter        bloom_filter_size bytes, absent when the size is 0
zone maps           64 bytes per 1024 rows, ceil(row_count / 1024) entries
null bitmap         ceil(row_count / 8) bytes, present only when null_count is not 0
encoded data        encoded_size bytes
padding             zero bytes up to the 64-byte boundary
```

The index entry for the segment records the segment start and the padded length. Offsets inside a segment are derived from the header and the file's row count:

```text
zones      = ceil(row_count / 1024)
null_start = 128 + bloom_filter_size + zones * 64
null_len   = null_count > 0 ? ceil(row_count / 8) : 0
encoded    = null_start + null_len, for encoded_size bytes
```

### Segment header, 128 bytes

| Offset | Size | Type | Field | Meaning |
| ------ | ---- | ---- | ----- | ------- |
| 0 | 4 | u32 | column_id | Catalog ordinal, shredded path id, or system column id |
| 4 | 1 | u8 | encoding_type | See Encoding types. An unknown value is rejected |
| 5 | 3 | bytes | reserved | Zero |
| 8 | 8 | u64 | raw_size | Bytes before encoding |
| 16 | 8 | u64 | encoded_size | Bytes of encoded payload |
| 24 | 8 | u64 | null_count | Nulls in the segment. A non-zero count means a null bitmap is present |
| 32 | 8 | u64 | cardinality | Distinct non-null values, counted up to the bloom threshold and capped there |
| 40 | 32 | bytes | min_value | Statistics slot, value bytes left-aligned and zero-padded on the right |
| 72 | 32 | bytes | max_value | Statistics slot. For variable-length values the prefix is rounded up so the bound stays conservative |
| 104 | 4 | u32 | data_checksum | `hash32` over the encoded payload only |
| 108 | 4 | u32 | header_crc | `hash32` over bytes 0 to 108 of this header, verified when the header is parsed |
| 112 | 8 | u64 | bloom_filter_offset | Segment-relative. 128 when a bloom is present, 0 otherwise |
| 120 | 4 | u32 | bloom_filter_size | Serialized bloom bytes, 0 when absent |
| 124 | 1 | u8 | is_sorted | Non-zero when the rows are sorted by this column's value |
| 125 | 3 | bytes | reserved | Zero |

### Zone maps

Zone maps split a segment into batches of 1024 rows (`ZONE_MAP_BATCH_SIZE`). One 64-byte entry per batch holds the batch minimum in bytes 0 to 32 and the batch maximum in bytes 32 to 64, in the same 32-byte statistics slot form as the segment header. The last batch may be partial and still occupies a full entry, so the zone map region size follows from the row count alone.

Slots compare under an order chosen by the column type: unsigned, two's complement, IEEE floating point, or lexicographic. A query skips every zone whose range cannot overlap its predicate without reading the zone's data.

### Null bitmap

One bit per row, present only when the segment has at least one null. A segment with no nulls omits the bitmap rather than writing zeros.

### Bloom filter

A split-block bloom filter answers whether a value might be present in a segment. Each probe touches exactly one 64-byte block, so a lookup costs one cache line.

Design constants:

| Constant | Value |
| -------- | ----- |
| `BLOOM_BITS_PER_ELEMENT` | 10, about 1% false positives |
| `BLOOM_HASH_COUNT` | 7 |
| `BLOOM_BLOCK_SIZE` | 64 bytes, 512 bits |
| `BLOOM_MIN_CARDINALITY` | 64 |

Sizing: `total_bits = elements * 10`, `num_blocks = ceil(total_bits / 512)` with a minimum of one block, and the bit array is `num_blocks * 64` bytes. The hash is the two-lane multiply-xor record hash. The block is chosen from one half of the 128-bit hash and the probe positions from the other half, stepping by an odd stride inside the block.

Which columns get a bloom is decided by the segment's bloom policy. The default policy builds one when the cardinality reaches 64 and the encoding is not dictionary, because a dictionary segment already carries exact membership. A policy can force a bloom for any segment with at least one value, or suppress it. Null rows are never inserted.

Serialized layout, 25 bytes of header followed by the bit array:

| Offset | Size | Field |
| ------ | ---- | ----- |
| 0 | 9 | Format stamp, magic `ZBLM` then u16 major and u16 minor, at version 1.0 |
| 9 | 4 | hash_count, u32 |
| 13 | 4 | num_blocks, u32 |
| 17 | 8 | num_elements, u64 |
| 25 | num_blocks * 64 | bit array |

The bloom carries no checksum of its own. It is covered by the segment it lives in. A reader rejects zero blocks, a zero hash count, a hash count above 14, or a bit array whose length is not `num_blocks * 64`. The in-place probe path used during scans answers "maybe present" on a malformed header rather than failing, so a damaged bloom prunes nothing and never hides a row.

## Segment index and trailer

The segment index is a run of 20-byte entries (`SEGMENT_INDEX_ENTRY_SIZE`), one per segment in the order the writer produced them:

| Offset | Size | Type | Field |
| ------ | ---- | ---- | ----- |
| 0 | 4 | u32 | column_id |
| 4 | 8 | u64 | Absolute offset of the segment start |
| 12 | 8 | u64 | Padded segment size |

The trailer is the last 20 bytes of the file (`FOOTER_SIZE`):

| Offset | Size | Field |
| ------ | ---- | ----- |
| 0 | 8 | segment_index_offset, a copy of the header field |
| 8 | 8 | Sentinel `ZYRCOL\0\0` |
| 16 | 4 | Index checksum, `hash32` over the index entries only |

The sentinel is not the file's identity, the envelope magic is. It tells a healthy tail from a truncated one.

The index checksum covers only the index region. Each segment's payload is covered by that segment's `data_checksum` and each segment header by its `header_crc`, so no read ever needs a whole-file pass.

## Opening a file

1. Read the first 512 bytes. Validate the envelope magic, `header_length` of 128, the version against the reader window, and the header checksum.
2. Refuse a `segment_index_offset` below 512, and a `segment_index_size` that is not a multiple of 20.
3. Read `segment_index_size + 20` bytes at `segment_index_offset`, the index and the trailer in one read.
4. Verify the trailer sentinel, verify that the trailer's index offset equals the header's, and verify the index checksum.
5. Parse the entries, build a lookup sorted by column id, and refuse a file that names one column twice.

That is three system calls: open, header read, index read. The file size is derived from the trailer position and never queried. Segment headers, zone maps, blooms and payloads are read on demand and memoized per reader. Payloads and blooms also live in a process-wide cache with a 64 MiB budget and a per-entry cap of one sixteenth of the budget.

## Encoding types

The `encoding_type` byte in the segment header takes one of these values:

| Id | Name | Stores |
| -- | ---- | ------ |
| 0 | Unencoded | The raw column bytes verbatim |
| 1 | Constant | A stored length and the single value once |
| 2 | BitPack | A bit width, the original value size, a base value, then frame-of-reference residuals at that width |
| 3 | Rle | Value and run-length pairs with varint run lengths. Refuses variable-length columns |
| 4 | Dictionary | A sorted dictionary of distinct values and a bit-packed code array of width `ceil(log2(dict_count))` |
| 5 | FastLanes | A frame-of-reference base, optional delta or delta-of-delta transform, a patched exception table, and bit-packed values |
| 6 | Alp | A factor and exponent pair that makes floats integral, the bit-packed integers, and an exception list for values that do not convert losslessly |
| 7 | Fsst | A 256-entry symbol table of frequent 1 to 8 byte substrings, the symbol-coded strings, and delta bit-packed offsets |

## System columns and visibility

Three hidden columns ride in every file in a reserved id range that catalog ordinals never reach:

| Id | Name | Meaning |
| -- | ---- | ------- |
| `u32::MAX` | `SYS_COL_ROWID` | Per-table monotonic row identity, survives merges. Marked as the primary key |
| `u32::MAX - 1` | `SYS_COL_XMIN` | Creating transaction id |
| `u32::MAX - 2` | `SYS_COL_SUPERSEDE` | Transaction that superseded the row version, 0 if never |

Each is an ordinary 8-byte unsigned segment with its own zone map, so per-zone visibility bounds come out of the `SYS_COL_XMIN` and `SYS_COL_SUPERSEDE` zone maps at no extra cost. The header's xmin range gives the same bound for the whole file so a snapshot can skip the file without opening a segment.

Shredded VARIANT paths use column ids from `1 << 20` up to `SYS_COL_SUPERSEDE - 1`.

Updates and deletes to rows that live in a `.zyr` do not rewrite the file. They append to a per-table `.zyrpatch` log that is folded into the base file at the next merge.

## Checksums

Every checksum in the file is `hash32` from the workspace checksum module, the 128-bit AES-lane hash folded to 32 bits. The streaming form, `Hasher` with `finish32`, produces the header and index checksums as the bytes are written.

| Checksum | Covers |
| -------- | ------ |
| Header checksum | File bytes 0 to 16 and 20 to 128. Not the checksum field itself, not the padding |
| Segment `header_crc` | Bytes 0 to 108 of that segment header |
| Segment `data_checksum` | The encoded payload of that segment |
| Index checksum | The concatenated index entries |

The bloom hash is a different function, the two-lane multiply-xor record hash, chosen because the AES hash measured 78% slower per probe.

## Versions

| Version | Header | Segment start | Written by |
| ------- | ------ | ------------- | ---------- |
| 1.0 | One full 16384-byte page | Next 16384-byte page boundary | Zyron 0.11.0 and earlier |
| 1.1 | 512 bytes | Next 64-byte boundary | Zyron 0.12.0 |

Nothing inside a segment changed between 1.0 and 1.1. The segment header, bloom, zone maps, null bitmap and payload are byte-identical, only their positions and the padding between them differ. A 1.0 file typically shrinks by more than four times when moved to 1.1.

Constants in `crates/zyron-storage/src/columnar/constants.rs`:

```rust
pub const ZYR_FORMAT_VERSION: FormatVersion = FormatVersion::new(1, 1);
pub const ZYR_FORMAT_VERSION_1_0: FormatVersion = FormatVersion::V1;
pub const ZYR_READER_WINDOW: VersionWindow = VersionWindow::new(ZYR_FORMAT_VERSION_1_0, ZYR_FORMAT_VERSION);
```

The registry entry for `ZyrColumnar` declares the migration policy eager, the migration not reversible, the binary version gate 0.12.0, and the retirement date 2027-03-01 for the 1.0 reader and its migration step.

The migration step `zyr_1_0_to_1_1` in `crates/zyron-storage/src/columnar/migrations/v1_0_to_v1_1.rs` takes the whole file and returns the whole file. It validates the header, the index size, the trailer and the index checksum before copying a byte, then copies only the bytes a reader ever touches from each segment, pads each to 64 bytes, rebuilds the index over the new offsets, and restamps the header at 1.1. A 1.1 input is refused, and a damaged index is refused rather than repacked. The fixture `crates/zyron-storage/src/columnar/fixtures/v1_0.bin` was written by Zyron 0.11.0 and pins the 1.0 layout.

A file outside the reader window is refused when the header is parsed, with `InvalidZyrFile`:

```text
columnar file is at format version 2.0, this binary reads 1.0..=1.1 and writes 1.1. Upgrade through a release that still reads 2.0 to move the file forward first
```

## Limits

| Limit | Bound | Set by |
| ----- | ----- | ------ |
| Columns declared in the header | 4,294,967,295 | `column_count` is a u32 |
| Columns addressable through the index | 214,748,364 | `segment_index_size` is a u32 over 20-byte entries |
| Rows per file | `u64::MAX` | `row_count` is a u64. In practice the zone map region grows 64 bytes per 1024 rows and each nullable column's bitmap one bit per row |
| Segment offset and size | 16 EiB | u64 index entry fields |
| Bloom bytes per segment | 4 GiB | `bloom_filter_size` is a u32 |
| Bloom hash count accepted on read | 1 to 14 | Twice `BLOOM_HASH_COUNT` |
| Statistics values | 32 bytes | `STAT_VALUE_SIZE`, longer values are truncated with the maximum rounded up |
| Header growth room | 384 bytes | Bytes 128 to 512 before `FILE_HEADER_SIZE` must change |
