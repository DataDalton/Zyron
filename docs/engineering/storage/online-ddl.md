# Online Heap DDL

How a schema change on a heap table completes without stopping the table, and why each step is where it is.

## Schema epochs

A heap tuple's bytes carry no column count and no type list. The row is a null bitmap of `ceil(columns / 8)` bytes followed by the values, and the only thing that says how many columns that bitmap covers is the schema the reader brings. Reading a row through the current column list is therefore correct only while the column list has not changed. A ninth column added to an eight-column table moves the bitmap from one byte to two, and every value after it lands one byte off.

The slot is what closes that. `TupleSlot` is 24 bytes:

```text
[offset u16][data_len u16][flags u16][schema_epoch u16][xmin u64][xmax u64]
```

Bytes 6..8 were spare and written as zero by every earlier writer. They now carry the epoch of the layout the row was encoded under. Nothing about the layout moved, which is why the heap page format bump from 1.0 to 1.1 has `no_body_change: true` and its migrator hands the page back untouched.

`TableEntry` records what each epoch was:

- `schema_epoch: u16` is the epoch every write stamps right now.
- `schema_epochs: Vec<EpochColumns>` holds, per epoch, the exact positional layout: one `PhysicalColumn { column_id, physical_type, fractional_digits, ordinal }` per encoded column, dropped columns included, in encoded order.
- `pre_stamp_columns: Vec<PhysicalColumn>` is the layout recorded at the version bump. A tuple stamped 0 predates stamping and reads through it.

A tuple whose epoch is neither 0 nor a recorded one is a corruption error naming the table, the page, the slot and the epoch. It is never read through some other layout: every column after the bitmap would land at the wrong offset, and a plausible-looking wrong row is worse than a refusal.

### What each column change does to the epochs

| Change | Epoch | Why |
| --- | --- | --- |
| `ADD COLUMN` | New epoch appended | The encoded shape gained a column, so the bitmap and every offset after it moved |
| `ALTER COLUMN ... TYPE <wider>` | New epoch appended | The encoded width changed |
| `DROP COLUMN` | Unchanged | The bytes stay exactly where they were; only the catalog stops naming the column |

### Absent values and dropped placeholders

`ColumnEntry` gained two fields.

`absent_value: Option<Vec<u8>>` is what a row that predates the column reads, encoded in the column's physical type. It is resolved once, when the column is added, which is why `ADD COLUMN c TIMESTAMP DEFAULT now()` gives every existing row the same instant rather than a different one per read. `None` means NULL.

`dropped: bool` marks a column gone from every user-facing listing while its bytes stay in place. Every enumeration a user, planner, wire row description, system view or default INSERT column list sees goes through `TableEntry::live_columns()`. The decoder and the tuple encoder do not: the decoder walks the placeholder to keep the cursor aligned, and the encoder still writes it so positions never shift under rows already on disk.

### Decoding

`EpochDecoder` (crates/zyron-executor/src/epoch_decode.rs) precomputes one `EpochPlan` per recorded epoch for one projection. A plan holds the bitmap length for that layout and one step per physical column:

- **Take** into a builder, with a widening: none, a signed or unsigned integer widened, a microsecond timestamp scaled to picoseconds, or a decimal rescaled to a larger declared scale.
- **Discard**, which advances the cursor and pushes nothing. A column dropped since the epoch, or one the projection did not ask for.

Plus one entry per current column the layout does not carry, which pushes that column's absent value.

Plans are indexed by epoch in a dense `Vec`, so the per-row cost is one bounds-checked lookup and a walk. A row of the current epoch does the same work the pre-phase decoder did.

Columnar segments carry their own column id list in the `.zyr` segment index, so a segment written before a column was added is missing that column rather than misaligned. The columnar scan fills it from the same `absent_value`, and a patch written after the fold still wins over it.

### Retirement

An epoch retires when no live tuple carries it. Vacuum is the only pass that visits every live tuple, so it is what can say so: `HeapPage::vacuum_in_slice` returns an `EpochCensus` holding the lowest epoch it saw above zero and whether it saw an unstamped tuple. `Catalog::retire_schema_epochs` drops every recorded layout below that minimum, and epoch 0 retires together with `pre_stamp_columns`, because they describe the same rows. A partial pass, one that stopped at the page limit, reports nothing: its minimum is not a minimum over the table.

The current epoch is never retired, whatever the census says, because the next write uses it. A pass that saw no live tuple at all changes nothing.

Vacuum, compaction and the columnar fold re-stamp what they rewrite with the current epoch and omit the placeholder bytes of dropped columns, which is how the space a dropped column occupied comes back. No forced rewrite exists.

## Publish, wait, scan, load, flip

An index build has to cover two sets of rows: what existed when it started, and what is written while it runs. Reading the first set and then registering the index leaves a gap, and a row written in that gap is never indexed. That is a wrong answer, not a slow one.

**Publish.** `create_btree_index` writes the entry with `IndexState::Building` and an empty tree goes into `server.btree_indexes`. From that instant every write that resolves its index set sees the index and maintains it.

**Wait.** This is the step the whole argument rests on. A transaction that resolved its index set *before* publication will not maintain the new index for the rest of its life, so its writes would fall in the gap. The build records `ProcArray::active_txn_ids()` at publication and waits until every one of them has ended. Afterwards, every writer still running resolved its index set after publication, so maintenance covers all of them.

The wait holds out the transactions the statement is itself running inside: a session's own open transaction under an explicit `BEGIN`, and the transaction an applier is replaying a schema change under. Neither resolved an index set for the table before publication, and both are transactions that cannot end until the build returns. `Session::open_txn_id` and `Session::apply_txn_id` carry them; the DDL dispatch sets the first on every statement and the replication applier sets the second.

**Scan.** One `Snapshot`, streamed in batches of at most `DEFAULT_BUILD_BATCH_ROWS` (65536). Heap pages in page order through `HeapPage::live_slot_in_slice`, columnar segments through the columnar scan with the patch overlay. Each batch's (key, locator) pairs go into `KeySorter`, which holds one run buffer and spills sorted runs to `<data_dir>/indexes/build-<index_file_id>/` when it fills. Nothing holds more than one batch of rows and one run buffer.

**Load.** `BTreeIndex::bulk_build_sorted` takes the merged sorted stream, merges it again with the tree's own in-order iterator (the keys concurrent maintenance has already placed), and writes leaves left to right, then each interior level from the level below, one pass per level. No key descends the tree. A key present in both streams lands once, and the tree's own entry wins the tie because it is the later write.

The swap of the new root is the one moment maintenance waits: `BTreeIndex::maintenance_guard` is taken for reading by the maintain path of a Building index only, and for writing across the root install and the catch-up that re-applies anything a writer placed while the levels were being written. A Ready index never touches that lock at all.

**Flip.** `DDL_INDEX_STATE` records Ready. Every index-selection path in the planner filters on `IndexState::Ready`, and a `SeqScan` that skipped a Building index whose key its predicate reaches carries the index's name in `deferred_index`, which `EXPLAIN` prints as `index_building`.

### Unique builds

The maintain path already takes a key lock and refuses a duplicate against the tree. The scan is what answers for the rows that predate the build: two live rows sharing a key fail it, and the failure names the key and both locators, drops the Building entry and its tree, and removes the spill directory.

### Other index kinds

Full-text, vector, spatial and hybrid indexes run the same publish, wait, flip sequence, because the race is in the sequence and not in the tree type. What differs is the fill: instead of a sorted run and a bulk load, each batch goes to `fill_search_indexes`, the same call an insert makes.

### REINDEX

Builds into a fresh index file id while the old tree keeps serving, then one catalog update points the entry at the new file. The old file's checkpoint is removed once the entry no longer names it.

## The shadow rewrite

`zyron_types::representation_compatible(from, to)` decides whether a type change can be answered by reading the old bytes a new way. It is true exactly for integer widening within one signedness, text and binary length widening, fractional-digit widening, and relaxing NOT NULL. The judgement is total over the source type's whole domain, never over the values a table happens to hold, so two tables with the same declaration behave the same way.

When it is false the rows are re-encoded, in crates/zyron-wire/src/shadow_rewrite.rs.

**Publish.** A hidden `TableEntry` named `zyron_shadow_<source id>_<source name>` with the new column type and its own heap files, plus a `ShadowSpec` in the source's `TableIndexSnapshot` maintenance list, the same list index maintenance walks. From that instant every insert, update and delete on the source applies to both heaps. `is_shadow_table` keeps it out of every listing.

**The hook.** `zyron_executor::shadow_write` mirrors an insert by casting the changed column and writing into the shadow heap, and records the source row's address against the shadow row's in the spec's row map. A delete looks the source address up and stamps the shadow copy's xmax. An update is the pair. A cast that fails aborts the writer's statement, because the alternative is a shadow that disagrees with the source and a swap that installs a table missing rows the writer was told it wrote.

**Wait.** As above, and for the same reason: a transaction that resolved its maintenance list before publication does not mirror.

**Copy.** The source streamed under one `Snapshot`, cast, written to the shadow at background priority. No sort, so no spill. A cast failure names the row's locator and value, drops the shadow and its files, removes the maintenance entry, and leaves the source untouched.

**Catch up.** The copy read one snapshot. A row committed after it is either mirrored by the hook, which covers every writer that resolved its maintenance list after publication, or written by one of the transactions the wait drained. The second set is found by reading the source again and taking every row the row map does not already name. The wait is what bounds it.

**Indexes.** Every B+tree the source declares is built on the shadow through the sequence above, so the swap installs a table that answers every access path the source did.

**Swap.** One `update_table` replaces the source's heap file ids, columns and epochs with the shadow's, and `seal_initial_epoch` starts its epoch history over because nothing on disk was written under any earlier one. Statements that already opened the old files finish on them; new statements resolve the entry and open the new ones.

## Constraints

`ADD CONSTRAINT` publishes the constraint with `validated: false` and enforces it on every write from that instant. The rows that predate it are checked after the same wait, and `validated` flips to true. `zyron_sys.core.constraints` reports the two states as `validating` and `valid`. A violating row removes the constraint and names the row: a rule the data does not satisfy never sits on a table unenforced.

## Recovery

`discard_incomplete_ddl` runs before a connection can be served, and it discards rather than resumes. A build cannot be resumed because the wait that made it correct ended with the process: rows a writer added after the crash were never maintained into the partial tree.

- A Building index: the catalog entry, the tree checkpoint and the `build-<file id>` spill directory all go, with one log line naming the index.
- An unvalidated constraint: removed, with one log line. It was enforced only by the process that published it.
- A shadow table: dropped with its files. The source is already correct, because the swap is the only step that touches it and the swap is one WAL record.

## Pressure

The scan, the sort, the load, the shadow copy and the catch-up run as a background workload class. `BuildPacer::between_batches` reads `PressureController::global()` between batches: anything above `ActuatorLevel::ReduceDop` yields, and three consecutive elevated readings stand the build down entirely, with `ddl_progress` reporting `paused_on_pressure` and the bottleneck and actuator that caused it. It resumes on its own. The check is a read of the controller's own state, never an admission request, so a build never takes the foreground path.

## Format and catalog registrations

- `FormatKind::HeapPage` at 1.1, reader window 1.0..1.1, a `retirement_date` six months past this release, gate 0.15.0. Migrator `heap_page_1_0_to_1_1` with `no_body_change: true`, fixture at crates/zyron-storage/src/heap/fixtures/v1_0.bin.
- `CatalogScanMigration { scan_name: "heap_pre_stamp_columns", one_way: true }` declares the catalog half. It is applied by the `zyron_sys.core.tables` evolution step, which fills `pre_stamp_columns` and epoch 1 from the row's own column list and is idempotent.
- `zyron_sys.storage.indexes` moves to schema version 2, whose step marks every existing index Ready.
- `FormatDocumentation` for heap page 1.1 carries the sentence `zyron_sys.storage.format_documentation` reports.

A page-resident fixture is not envelope framed, so the release check reads its identity from the page header's stamp rather than from the file's first bytes.
