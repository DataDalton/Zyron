# Format Agility, Developer Notes

Implementation reference for Zyron's on-disk file format versioning and automatic migration substrate. Customer-facing documentation lives in [`business/storage/format-agility.md`](../../business/storage/format-agility.md).

## Core rule

No legacy code. Cutover plus automatic migration, not coexistence with compatibility shims. On upgrade, existing state is rewritten to the new form, and after the migration window closes the old reader path is deleted. The release check enforces this: `zyron-ctl release verify` fails a candidate release when a format bump ships without its migrator or its fixture, and again when a reader still carries a version past its retirement date.

## Envelope

Every persistent file Zyron writes begins with a 20-byte envelope header:

```text
[0, 4)    magic, the 4-byte identifier for the format kind
[4, 6)    version_major, u16
[6, 8)    version_minor, u16
[8, 12)   header_length, u32, 20 plus the per-format extension length
[12, 16)  flags, u32
[16, 20)  header_checksum, u32
[20, header_length)   per-format extension
[header_length, len-4)  body
[len-4, len)  footer_checksum, u32
```

The header checksum covers bytes 0 to 16 and the extension, not the checksum field itself. The footer checksum covers the body. Both are `hash32` from the workspace checksum module. The flag bits are `COMPRESSED`, `ENCRYPTED` and `DOWNGRADE_WRITE` in the low byte, which the substrate owns, with the high three bytes reserved to the format.

## Framings

A format's framing says how much of the envelope it carries and how a migrator for it is shaped. `FormatKind::framing()` is the one mapping.

| Framing | Header | Footer | Kinds |
| ------- | ------ | ------ | ----- |
| Envelope | 20 bytes | 4 bytes | WalSegment, Snapshot, Checkpoint, MvccClog, StatisticsFile, LakeIndex, RaftLog, and the other container files |
| OwnTrailer | 20 bytes | 0 | ZyrColumnar, LakeManifest |
| Stamp | 9 bytes | 0 | HeapPage, BTreeInternal, BTreeLeaf, Fsm, BloomFilter, Toast |
| Text | 0 | 0 | ZyronTomlConfig, BackupArchive |
| RecordTag | 1 byte | 0 | AuditHashChain, ReplicationApplyLog, DeletePredicate, LakeTransactionLog |

### Formats that own their trailer

The columnar file (`.zyr`) and the lake manifest carry the envelope header and then a trailer of their own instead of the envelope footer checksum. The `.zyr` ends in a segment index whose checksum covers the index alone, so a reader verifies each segment by that segment's own checksum and never reads the whole file to open it. The manifest ends in a CRC over the whole buffer followed by a sentinel.

The substrate treats such a file differently in three places. Opening it reads only the header for kind and version and leaves the trailer to the format's reader. A migrator step takes and returns the whole file rather than a body, because the format's checksums cover the header the envelope path would otherwise restamp, and `OpenedFile::reencode` returns the migrated bytes verbatim. `Framing::migrates_whole_file()` is true only for this framing. A step for such a format cannot be marked `no_body_change`, the registry refuses that registration with `RestampOnOwnTrailer`.

## Per-record version tags

Records inside a container cannot use the envelope. A WAL entry cannot afford a 20-byte header, and a segment written across a rolling upgrade holds records of more than one version, so a version on the container could not describe it. Each record carries its own tag instead.

`crates/zyron-common/src/format/record_version.rs`:

```text
RecordVersion(u8)         inline tag, usable range 1 through 254
WideRecordVersion(u16)    wide tag, usable range 1 through 65535
```

Two byte values are reserved and never name a version:

| Value | Constant | Meaning |
| ----- | -------- | ------- |
| 0 | `RECORD_VERSION_ABSENT` | The slot was never written. A zero-filled region reads as absent rather than as version zero |
| 255 | `RECORD_VERSION_WIDE_ESCAPE` | The version did not fit one byte. A `WideRecordVersion` carries it instead |

`RecordVersion::new` returns `None` for both, so neither can be handed out as an ordinary version, and `RECORD_VERSION_MAX_INLINE` is 254.

### Why the escape is reserved

Without it the one-byte tag is a dead end. A stream reaching 255 would have no value left to say the layout changed again, and widening the tag would need a flag day where every reader is replaced at once. With it the change announces itself. A reader that predates it sees the escape and refuses the record by name through `EnvelopeError::BadRecordVersion`, rather than misparsing the byte as version 255 and reading the record with the wrong decoder.

Nothing writes the escape today. Every stream below is at version 1.

### Reading a tag

Two entry points, chosen by whether the caller can read the bytes behind the tag:

- `RecordVersion::read(bytes)` reads the single byte and refuses the escape. Used where the layout has no room behind the tag.
- `RecordTag::read(bytes)` resolves the escape by reading the two bytes after it. It returns the tag and the bytes consumed, 1 for an inline tag and 3 for a wide one. `RecordTag::as_format_version` widens either into a `FormatVersion` so inline and wide tags order against each other across the change.

### Where the tag lives per stream

| Stream | Constant | Position |
| ------ | -------- | -------- |
| WAL entry | `WAL_RECORD_VERSION` | Fixed header byte 25, see caveat |
| Raft log entry | `RAFT_ENTRY_RECORD_VERSION` | Byte 4, after the `ZRAF` magic |
| Replication apply entry | `APPLY_RECORD_VERSION` | Byte 0 of the chunk header |
| Audit chain event | `AUDIT_RECORD_VERSION` | A field on the compliance log row, folded into the entry hash |

The lake transaction log declares `Framing::RecordTag` but its commit record carries a 4-byte magic and a 4-byte `FormatVersion` prefix rather than a one-byte tag.

**WAL caveat.** The WAL record header is a fixed 28 bytes with the tag at offset 25 and `payload_len` immediately behind it at 26. There is no room to place two more bytes after the tag without repacking the header, which would cost four bytes per record after realignment. The release that spends the escape defines where the WAL's wide version sits. Until then the record reader reports the escape as a layout it has no reader for and names the upgrade path, rather than reporting it as damage.

## Types

`crates/zyron-common/src/format/registry.rs` and `version.rs`:

```rust
pub struct FormatVersion { pub major: u16, pub minor: u16 }

pub struct VersionWindow { pub oldest: FormatVersion, pub newest: FormatVersion }

pub struct FormatRegistration {
    pub kind: FormatKind,
    pub writer_current_version: FormatVersion,
    pub reader_supported_versions: VersionWindow,
    pub migration_policy: MigrationPolicy,   // Eager, Lazy, Coexist
    pub migration_reversible: bool,
    pub binary_version_gate: &'static str,
    pub deprecation_status: DeprecationStatus,
    pub retirement_date: Option<&'static str>,
    pub downgrade_write_supported: bool,
    pub notes: &'static str,
}

pub type MigrateFn = fn(&[u8]) -> Result<Vec<u8>, String>;

pub struct FormatMigrator {
    pub kind: FormatKind,
    pub from: FormatVersion,
    pub to: FormatVersion,
    pub reversible: bool,
    pub forward: MigrateFn,
    pub backward: Option<MigrateFn>,
    pub no_body_change: bool,
    pub description: &'static str,
}

pub struct FormatFixture {
    pub kind: FormatKind,
    pub version: FormatVersion,
    pub bytes: &'static [u8],
    pub path: &'static str,
}
```

Each type is gathered with `inventory::submit!`. A migrator's `forward` and `backward` take and return the body for an envelope-framed format and the whole file for an own-trailer format. For an envelope format that only restamps the version, set `no_body_change` and omit the function.

## Registry validation

The registry rejects an inconsistent set at load, which is what the startup gate runs. The rules:

- Exactly one registration per format kind, and every one of the format kinds has a registration.
- The writer version is inside the reader window.
- A migrator for every adjacent step from the window's oldest version up to the writer version. A step with no migrator fails with `MissingMigrator`, naming the `migrations/vM_m_to_vM_m.rs` file to add.
- A fixture for every version in the window below the writer version, failing with `MissingFixture`, naming the `fixtures/vM_m.bin` file to add.
- A migrator marked `reversible` carries a `backward` function unless it is `no_body_change`.
- Migrator steps are adjacent, one minor apart or one major with the minor reset to zero. The substrate chains them.
- An own-trailer step cannot be `no_body_change`, it must rewrite the file whole.
- A window holding more than one version declares a `retirement_date` for the oldest, and any `retirement_date` is an ISO 8601 calendar date.

There is no cap on how many versions a window may hold.

## Runtime behavior

On open, `migration::open` peeks the header for the kind, then `open_as` selects a reader path from the version. A version equal to the writer version is the current path with no migrator scan. A version inside the window but below the writer version plans a chain of adjacent migrators. A version outside the window fails closed with `EnvelopeError::UnknownVersion`, naming the versions the binary reads. A version inside the window but above the writer version, which is an older node reading a newer peer's file, also fails closed.

`OpenedFile::reencode` writes the migrated bytes back in the file's framing: verbatim for an own-trailer format, wrapped in a fresh envelope header and footer for an envelope format.

The migration policy decides when a migrated file is persisted. Eager rewrites it during a background sweep budgeted per configuration. The sweep decides from the envelope header alone whether a file is behind, so a file already at the writer's version is never read past its first bytes, and a file another subsystem is still writing is never opened whole. Lazy rewrites it the next time the owning subsystem writes the file, which is the policy of a format that rolls forward by itself, such as a WAL segment that is written once at the current version and retired by the next checkpoint. Coexist never rewrites it, both versions persist, which is what immutable historical data such as snapshots and audit chains use.

## Format registry view

`zyron_sys.storage.format_registry` projects the live registry, one row per format kind, with these columns: `format_kind`, `magic`, `owner`, `writer_current_version`, `reader_oldest_version`, `reader_newest_version`, `migration_policy`, `migration_reversible`, `binary_version_gate`, `deprecation_status`, `retirement_date`, `downgrade_write_supported`, `migrator_count`, `fixture_count`, `notes`. The companion `zyron_sys.storage.format_documentation` has one row per readable version with its framing, header and footer sizes, and layout. In-progress migrations show in `zyron_sys.storage.format_migrations`.

## Developer workflow: bumping a format version

Say the heap page format goes from 3.0 to 4.0 because a field widens. The migrations and fixtures for a format live in folders beside it, one file per step and one file per still-readable version, so retiring a version is deleting its step file, its fixture, and one `mod` line.

1. Bump the version constants and widen the reader window in `crates/zyron-common/src/page.rs`, where `HEAP_PAGE_FORMAT_VERSION` lives:

   ```rust
   pub const HEAP_PAGE_FORMAT_VERSION: FormatVersion = FormatVersion::new(4, 0);
   pub const HEAP_PAGE_FORMAT_VERSION_3_0: FormatVersion = FormatVersion::new(3, 0);
   pub const HEAP_PAGE_READER_WINDOW: VersionWindow =
       VersionWindow::new(HEAP_PAGE_FORMAT_VERSION_3_0, HEAP_PAGE_FORMAT_VERSION);
   ```

2. Add the new reader for the 4.0 layout, keeping the 3.0 reader until its retirement date.

3. Add the step file `crates/zyron-storage/src/heap/migrations/v3_0_to_v4_0.rs`, declared with `mod v3_0_to_v4_0;` from `migrations/mod.rs`. The step file carries the function, its `FormatMigrator` submission, and the `FormatFixture` submission for the version it reads:

   ```rust
   static FIXTURE_3_0: &[u8] = include_bytes!("../fixtures/v3_0.bin");

   inventory::submit! {
       FormatFixture {
           kind: FormatKind::HeapPage,
           version: HEAP_PAGE_FORMAT_VERSION_3_0,
           bytes: FIXTURE_3_0,
           path: "crates/zyron-storage/src/heap/fixtures/v3_0.bin",
       }
   }

   inventory::submit! {
       FormatMigrator {
           kind: FormatKind::HeapPage,
           from: HEAP_PAGE_FORMAT_VERSION_3_0,
           to: HEAP_PAGE_FORMAT_VERSION,
           reversible: false,
           forward: heap_3_0_to_4_0,
           backward: None,
           no_body_change: false,
           description: "column count widened to u32",
       }
   }

   pub fn heap_3_0_to_4_0(body: &[u8]) -> Result<Vec<u8>, String> {
       // read the 3.0 body, return the 4.0 body
   }
   ```

4. Add the fixture `crates/zyron-storage/src/heap/fixtures/v3_0.bin`, a file the release that wrote 3.0 actually produced, not one round-tripped through the current writer.

5. Update the format's `FormatRegistration` in `crates/zyron-storage/src/format.rs` with the new writer version, the widened reader window, and a `retirement_date` for 3.0.

The release check verifies the migrator and fixture are present and fails the release if either is missing.

## Worked examples in the tree

Two formats carry a live migration today, both own-trailer, both forward-only.

The `.zyr` columnar file moved 1.0 to 1.1 in `crates/zyron-storage/src/columnar/migrations/v1_0_to_v1_1.rs`. The step repacks segments onto a 64-byte alignment behind a 512-byte header, in place of a page of padding per segment. Its fixture is `columnar/fixtures/v1_0.bin`, written by 0.11.0. See [formats/zyr-spec.md](formats/zyr-spec.md).

The lake manifest moved 2.0 to 2.1 in `crates/zyron-lake/src/manifest/migrations/v2_0_to_v2_1.rs`. The step decodes and re-encodes, adding an exact column sum behind a presence flag. Its fixture is `manifest/fixtures/v2_0.bin`.

Both declare the migration policy eager, the migration not reversible, and the retirement date 2027-03-01 for the reader of the older version. The user's stance is old to new only, no backward function and no downgrade for these steps.

## Retention lifecycle

- A reader is carried for the current version and the versions in its window. Widen the window when a version is added, narrow it when the oldest retires.
- A migrator moving files from the oldest supported version into the window is kept as long as that version is in the window.
- A major version boundary is the place to consolidate prior-major migrations into a single step.
- The deprecation registry keeps a metadata-only record, no code.

When a version passes its `retirement_date`, `zyron-ctl release verify` fails while its reader, migrator, or fixture is still in the tree, so the decision to delete old code is a date, not a judgment call.

## Unknown version handling

A version outside the supported window fails closed with a structured error rather than a silent misread. The columnar reader phrases it directly:

```text
columnar file is at format version 2.0, this binary reads 1.0..=1.1 and writes 1.1. Upgrade through a release that still reads 2.0 to move the file forward first
```

## CLI

- `zyron-ctl format inspect <file>` reports the kind, version, framing, and integrity of one file.
- `zyron-ctl format migrate --format <kind> [--to <version>] --path <dir>` moves every file of a kind under a directory to the version this binary writes, refusing any other target.
- `zyron-ctl format verify --path <dir>` checks every file in a directory against its registered format.
- `zyron-ctl release verify [--today <YYYY-MM-DD>]` checks every format bump in a candidate release for a migrator, a fixture, and a registry entry, and reports any reader past its retirement date.

## Related

- [formats/zyr-spec.md](formats/zyr-spec.md), the `.zyr` columnar file format.
- [formats/zyridx-spec.md](formats/zyridx-spec.md), the `.zyridx` index checkpoint format.
- [../operations/auto-upgrade.md](../operations/auto-upgrade.md), upgrade orchestration internals.
- [../security/signature-agility.md](../security/signature-agility.md), scheme rotation internals.
- Business-facing counterpart: [`business/storage/format-agility.md`](../../business/storage/format-agility.md).
