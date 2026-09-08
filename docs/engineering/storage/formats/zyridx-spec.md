# .zyridx Index Checkpoint File Format Specification

Format version 11.1. The running binary writes 11.1 and reads back to 11.0, moving an 11.0 file forward through the registered migration. A checkpoint older than that is refused and the index is rebuilt from the write-ahead log instead.

## Overview

A `.zyridx` file is a checkpoint of one B+Tree index, every leaf entry of the tree at a given write-ahead log position, laid out as one fixed-stride run of entries behind a shared key prefix. Loading a checkpoint rebuilds the leaf chain and the internal levels in memory, so recovery replays only the log written after the checkpoint position.

An index key is a value followed by a seventeen byte order-preserving suffix naming the row it points at. The file stores the value and a locator of seven bytes for a heap row, which is the same address in ten fewer bytes, and the suffix is rebuilt when the file loads. The shared prefix is therefore measured over values, values being the part keys have in common and the suffix being unique per entry.

The file carries the 20-byte envelope header every persistent Zyron file starts with, a 20-byte checkpoint extension, the body, and a 4-byte envelope footer checksum. The registry reports the kind `Checkpoint` with the envelope framing.

All multi-byte integers are little-endian.

## File layout

```text
[0, 40)                 header, envelope prefix then checkpoint extension
[40, 40 + P)            value prefix, P = prefix_len bytes
[.., + N * (S + W))     entries, each an S-byte value tail then a W-byte locator
last 4 bytes            footer checksum over the whole body
```

With `N` the entry count, `V = key_len - 17` the value length, `S = V - prefix_len` and `W = value_width`, the file size is `40 + P + N * (S + W) + 4`. An empty checkpoint is exactly 44 bytes.

The seventeen subtracted from `key_len` is the suffix naming the row, which the file does not carry.

## Header, 40 bytes

### Envelope prefix, bytes 0 to 20

| Offset | Size | Type | Field | Value |
| ------ | ---- | ---- | ----- | ----- |
| 0 | 4 | bytes | magic | `ZCPT`, the magic allocated to `FormatKind::Checkpoint` |
| 4 | 2 | u16 | version_major | 11 |
| 6 | 2 | u16 | version_minor | 1 |
| 8 | 4 | u32 | header_length | 40. The loader refuses any other value |
| 12 | 4 | u32 | flags | Always 0. No checkpoint is compressed or encrypted |
| 16 | 4 | u32 | header_checksum | Covers bytes 0 to 16 and 20 to 40 |

### Checkpoint extension, bytes 20 to 40

| Offset | Size | Type | Field | Meaning |
| ------ | ---- | ---- | ----- | ------- |
| 20 | 8 | u64 | checkpoint_lsn | Write-ahead log position the checkpoint was taken at |
| 28 | 4 | u32 | entry_count | Index entries in the file |
| 32 | 2 | u16 | key_len | Length of every whole key, value and row-naming suffix together, taken from the first entry of the first leaf |
| 34 | 2 | u16 | prefix_len | Bytes of value shared by every entry and stored once |
| 36 | 2 | u16 | value_width | Locator bytes behind each value, 7 or 17 |
| 38 | 2 | bytes | reserved | Zero |

The header stores no root page and no page count. Both are recomputed when the leaves are rebuilt.

Every key in one file has exactly `key_len` bytes, so the format carries no per-entry key length. The leaf page it rebuilds into carries none either, the slot pointing at an entry holding its length.

## Body

### Value prefix

The writer compares the first key of the first leaf with the last key of the last leaf byte by byte, stopping at the value length rather than the whole key length. Because the leaf chain is sorted, those two keys bound every key in the tree, so the bytes they share are shared by every entry. That prefix is written once. It is empty when the tree has a single leaf or when values are empty.

The comparison stops at the value because the bytes past it are the suffix naming the row, which is unique per entry and shares nothing.

### Entries

`entry_count` entries in leaf-chain order, which is ascending key order. Each is the value past the shared prefix, `key_len - 17 - prefix_len` bytes, followed by the `value_width` byte locator naming its row. There is no offsets table, entry `i` starts at `entries_start + i * (suffix_len + value_width)`.

The locator is read out of the trailing suffix of the key the leaf holds, and the load writes it back into the rebuilt key as that suffix. Its tag byte comes first:

| Width | Tag | Layout |
| ----- | --- | ------ |
| 7 | 3, heap narrow | u32 page number, u16 slot |
| 17 | 0 heap, 1 columnar, 2 lake | two u64 fields |

The width is uniform across the file. The writer takes it from the first entry and checks every later entry against it. If a wider locator appears part way through, the writer discards the buffer, switches the whole file to width 17, and starts over from the first leaf. A narrow locator in a wide file is re-encoded in the wide form.

## Footer, 4 bytes

The last four bytes are a u32 checksum over the whole body, bytes 40 to the footer. It does not cover the header, which has its own checksum.

Both checksums are `hash32` from the workspace checksum module, the 128-bit AES-lane hash folded to 32 bits. The writer folds the body in as it is written, prefix first then every entry span in file order, with the streaming `Hasher`. The loader recomputes it in one pass over the body.

## Write procedure

The index takes its root-change lock so the root and the height cannot move during the write. Leaf splits below the root still proceed, the checkpoint position covers them on replay.

1. Descend the leftmost child pointers to the first leaf, then walk the leaf chain collecting each leaf's page number and slot count, the total entry count, and the key length and value width of the first entry.
2. Compute the key prefix and the file size, and allocate the whole file image without zero-filling it. Every byte of the image is written before it is read.
3. Stamp the header and the key prefix.
4. Create the target file and set its length up front, so no writer ever extends it.
5. Gather every leaf's entries into the body, in runs of about 4 MiB. Below 32 MiB of body the calling thread writes each run itself. Above that, up to 8 writer threads, one per 4 MiB and never more than the machine has cores, take finished runs from a queue and write them with positioned writes, which is what lets them share one file handle without disturbing each other's position.
6. Write the header span and the footer with positioned writes.
7. Call `sync_all` on the file when the index is configured to fsync.

The caller writes to `index_{file_id}.zyridx.tmp` under the checkpoint directory and renames it over `index_{file_id}.zyridx` once the write returns, then records the checkpoint position. A crash during the write leaves the previous checkpoint intact.

An empty tree writes the 44-byte empty checkpoint with a zero entry count, zero widths, and the checksum of an empty body.

## Load procedure

1. Read the file size, reserve a buffer for it without zero-filling, and read the file with positioned reads. Below 4 MiB one read on the calling thread, above that one reader per 2 MiB up to 8, each opening its own file handle.
2. Refuse a file shorter than 44 bytes.
3. Decode the envelope header, which checks the magic against the registered kinds and verifies the header checksum.
4. Refuse a kind other than `Checkpoint`, a `header_length` other than 40, and a version other than 11.1. An 11.0 file reaches this point already moved forward by the registered migration.
5. Parse the extension. Refuse a `value_width` other than 7 or 17 when the entry count is not zero.
6. Recompute the expected size from the extension and refuse a shorter file.
7. Verify the footer checksum over the body.
8. Rebuild the leaves, then the internal levels.

Every refusal is `RecoveryFailed`. The index open path treats a missing or refused checkpoint the same way, it starts from an empty tree and lets the caller replay the whole log.

### Leaf rebuild

Leaf geometry follows from the entry size `key_len` and the page size of 16384 bytes. A leaf entry is the whole key and nothing else, its length held by the slot and the row it names by its own trailing suffix, which the rebuild writes from the locator stored behind each value. Every rebuilt leaf but the last is full, so leaf `k` holds entries from `k * max_entries_per_page`, and a run of leaves can be rebuilt knowing only its starting leaf index. The rebuild runs on up to half the machine's cores, one run per 64 or more leaves.

Pages come from the store uninitialized. The rebuild writes every byte of every page, including the free space between the slot array and the first entry, so a recycled buffer can never leak another page's bytes into a leaf that may reach disk. A test loads the same checkpoint twice into separate stores and asserts the pages are byte-identical with all-zero free space.

Each leaf's next-leaf pointer is set to the following page, the last leaf's to the end marker.

### Internal levels

A single leaf is the root, height 1. Otherwise internal pages are built level by level from the first key of each child, filling each internal page to three quarters of its usable space so the tree has room to grow without immediate splits. The last remaining page is the root, and the height is the number of levels built plus one.

## Recovery integration

`BTreeIndex::open` loads `{checkpoint_dir}/index_{file_id}.zyridx` and returns the checkpoint position so the caller replays the log from there. `TRUNCATE` and `REINDEX` delete the file when they rebuild an index, so recovery cannot reload stale keys. Backups include `.zyridx` among the data extensions.

Write-ahead log segments become deletable when the checkpoint coordinator finishes a checkpoint. Segments whose id is strictly less than the segment holding the global minimum checkpoint position across all tables are removed, subject to retention hooks that can hold the bar lower. The segment containing the position is kept because recovery replays from that offset.

## Version

Constants in `crates/zyron-storage/src/format.rs`:

```rust
pub const CHECKPOINT_FORMAT_VERSION: FormatVersion = FormatVersion::new(11, 0);
```

The registry entry for `Checkpoint` declares a reader window of 11.0 through 11.1, the migration policy eager, the migration one way, and the binary version gate 0.15.0. Eager here means the next checkpoint rewrites the file in full, which is how every checkpoint is written anyway. The step is one way because a checkpoint is derived from the tree it describes, so moving back down is rebuilding it rather than transforming the file. The date 11.0 retires on is `CHECKPOINT_11_0_RETIREMENT` in `crates/zyron-storage/src/format.rs`, six months past the release that shipped 11.1.

A checkpoint outside that window is refused when the header is parsed:

```text
checkpoint is at format version 10.0, this binary loads 11.1. A 11.0 file is moved forward by the registered migration before it reaches this point
```

## Limits

| Limit | Value | Set by |
| ----- | ----- | ------ |
| Entries per file | 4,294,967,295 | `entry_count` is a u32 |
| Key length | 256 bytes | `MAX_KEY_SIZE`, enforced at insert. The field itself is a u16 |
| Value width | 7 or 17 | The two row locator forms |
| Checkpoint position | `u64::MAX` | `checkpoint_lsn` is a u64 |
| Entries per rebuilt leaf | `(16384 - 56) / (key_len + 4)` | Page size, page and leaf header sizes, slot size. 563 for an 8-byte value, whose whole key is 25 bytes |
| Writer threads | 8 | One per 4 MiB of body, never more than cores |
| Reader threads | 8 | One per 2 MiB of file, never more than cores |
| Rebuild threads | Half the cores | One run per 64 leaves at least |
