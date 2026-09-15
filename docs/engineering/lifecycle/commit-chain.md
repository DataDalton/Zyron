# The Commit Chain

How a verifiable table's hash chain is built, stored, replicated and walked.
The user-facing counterpart is
[../../business/security/verifiable-tables.md](../../business/security/verifiable-tables.md).

Everything here lives in `crates/zyron-lifecycle/src/verify.rs` and
`crates/zyron-lifecycle/src/verify/anchor.rs`, with the paths that drive it in
`crates/zyron-wire/src/verify_dispatch.rs` and, on a consensus group, in
`crates/zyron-server/src/replication.rs`.

## Per commit, not per row

One chain entry covers one committing transaction's writes to one table. The
alternative, one entry per row, was not taken for three reasons.

Storage is the obvious one. A per-row chain over a table taking a thousand
rows a second costs more than the rows. Per commit it is a fixed 96 bytes
however many rows the commit wrote.

The serial section is the real one. A chain is inherently serial: an entry
links to the entry before it, so producing one means reading the current head.
Per row that serialisation happens a million times for a million-row load. Per
commit it happens once, and the expensive half, hashing the rows, happens
outside it.

The third is that a commit is the unit the question is asked about. "Was this
row changed" and "was this commit's set of rows changed" have the same answer
whenever the second is no, and the second is what a transaction actually
promised.

## Where the work happens

Hashing the rows runs on whatever thread the statement runs on, as the rows
are written:

```text
InsertOperator::next
  ctx.chain_sink.open_write(table)     the registry's answer, and the cursor
  heap_file.insert_batch_with_cursor   rows land above the ones before them
  ctx.chain_sink.rows(...)             one SHA-256 pass, no lock
```

`CommitChainSink` is declared in `zyron-executor/src/context.rs` and
implemented in `zyron-wire/src/verify_dispatch.rs` over
`PendingChainWrites`, which holds one running `Sha256` per table per
transaction. A commit of a million rows holds one hash state, not a million
rows. Every write to every table reports to the sink, chained or not: what a
transaction touched is what the commit holds against the fence a table was
chained at.

Whether a table's rows are chained is the registry's answer
(`ChainRegistry::chained`), not the catalog entry a statement resolved. A
table becomes verifiable for the write path at one instant on a node, and a
statement planned against an older entry is chained all the same.

Linking runs at commit, under the table's chain lock:

```text
Connection::commit_with_advances
  log_stream_advances(...)             into the transaction's log chain
  take_commit_chains(...)              one hash per table, or a refusal
  link_commit_chains(...)              read head, hash the fields, push
  txn_manager.commit(...)              commit record, then durability
  publish_commit_chains(...)           the record reaches the file's buffer
```

The lock is held for a head read, one SHA-256 over the entry's own fields, and
a push onto a pending list. It is not held across the durability wait, and
publishing is a copy into the file's write buffer rather than a system call.
Sixteen writers therefore hash in parallel and queue only for the link.

## The record

96 bytes, fixed, no framing of its own:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 8 | `commit_version` |
| 8 | 8 | `commit_ts` |
| 16 | 8 | `txn_id` |
| 24 | 4 | `row_count` |
| 28 | 2 | `algorithm_id` |
| 30 | 1 | flags, bit 0 set on the genesis entry |
| 31 | 1 | reserved, zero |
| 32 | 32 | `rows_hash` |
| 64 | 32 | `entry_hash` |

`prev_hash` is not stored. It is the previous record's `entry_hash`, and the
genesis entry's is zero. Storing it would cost a third of the record to say
what the record before it already says.

`table_id` is not stored either: the chain is per table, at
`<data_dir>/verify/<table_id>.zvch`, behind a 20-byte envelope header
(`FormatKind::VerifiableCommitChain`, magic `ZVCH`). File length minus the
header, divided by 96, is the commit count, and the last record is the head.
That is what makes the per-commit storage figure in the user guide exact
rather than approximate. A file whose length is not a whole number of records
was cut by a stop between a write and its flush; the reopen cuts the partial
tail off, so the next record lands on a record boundary.

`algorithm_id` is per record rather than read from the table, so an entry
states what hashed it. It is a `SchemeId` from the signature scheme registry,
where SHA-256 sits under `SchemeCategory::Hash`. Registering the hash in the
same registry as the signature schemes means one lookup answers what a stored
artifact names, whatever kind of artifact it is.

`row_count` is a `u32`. A commit above `MAX_COMMIT_ROWS` is refused rather
than recorded short, because an entry that undercounts what its commit wrote
is worse than no entry.

`commit_version` is read under the chain lock and never falls below the
head's, so versions rise along the chain. On a node that leads no group it is
the log position, on a member of a group the index of the entry that carried
the commit. A `FROM VERSION` or `TO VERSION` clause resolves to positions by
binary search over the file, one record read per probe.

## What the link covers

`entry_hash` is SHA-256 over `prev_hash`, `table_id`, `commit_version`,
`rows_hash`, `row_count`, `commit_ts`, `algorithm_id` and the flags byte.

It is not over `txn_id`. The transaction id is the stamp a member's own rows
carry and so the key a verification on that member reads them back by, and
every member of a group stamps its rows with an id of its own. Leaving it out
of the link is what lets every member hold the same chain over the same rows.
An entry whose id was edited still fails a verification: the rows read back
by the edited id are not the rows the entry recorded, so their hash or their
count does not match.

### How a digest is computed

The link, a row's digest for the genesis set, and any other one-shot digest
in the module go through `sha256_of`, which lays the standard padding out by
hand and drives the blocks onto `sha2::compress256` with the state in a heap
slot held per thread. The streaming hashers (`RowsHasher`, `SetHasher`) use
the crate's `Sha256` as they are, which keeps its state wherever the hasher
lives.

The heap slot is the point. The compression stores its result with vector
stores and a digest is read back from it a word at a time, and on the
processors this runs on that read costs two orders of magnitude more from a
stack local than from a heap slot. A digest per chain entry through a stack
local made the link, and so the serial section of every commit and the
link check of every walk, cost what a handful of disk reads cost. The
streaming hashers were never affected because the rows pass holds them in a
`Vec`. The benchmark prints the link hash beside the crate's own one-call
digest over the same bytes as the record of the difference.

## Canonical encoding

A row is hashed as:

```text
schema_epoch (u16 LE) || row_length (u32 LE) || row bytes
```

The row bytes are the stored form, exactly as the heap holds them. No padding,
no re-serialisation, no schema-dependent ordering beyond the column order the
epoch records.

The length is in the hash so two rows cannot hash as one row of their
concatenated bytes.

### Schema epochs

The epoch is part of what is hashed, which is what makes a schema change
harmless to earlier entries. A column added between two commits mints a new
epoch (see [../storage/online-ddl.md](../storage/online-ddl.md)); rows written
before it carry the old epoch and rows after it carry the new one. A
verification reads each row back with the epoch it carries and hashes it under
that epoch, so both commits verify against the entries they produced.

Without the epoch in the hash, a table that gained a column would have every
entry before the change fail, and the failure would be indistinguishable from
tampering.

### Which schema changes a chain survives

The hash covers stored bytes and the per-row epoch stamp, never decoded
values, and an epoch stamp is written into the tuple header at insert time.
That decides the question for each kind of change:

| Change | Stored bytes | Chain |
| --- | --- | --- |
| ADD COLUMN | Untouched, a new epoch is minted | Survives |
| DROP COLUMN | Untouched, the column keeps a placeholder | Survives |
| ALTER COLUMN TYPE, representation compatible | Untouched, widened as read | Survives |
| ALTER COLUMN TYPE, needing a rewrite | Every row re-encoded | Refused |

The compatible path (`ddl_dispatch.rs`, the branch after
`representation_compatible`) changes `type_id`, `max_length` and
`fractional_digits` on the catalog entry and mints an epoch. Rows already
written keep both their bytes and the epoch they were stamped with, so a
verification reproduces their hashes exactly.

The incompatible path runs a shadow rewrite, which re-encodes every row into
a new heap. That is the same thing to a chain as an UPDATE of every row, so
it answers to `refuse_write_locked` alongside DROP and TRUNCATE. Without
that refusal an ordinary `ALTER TABLE ... ALTER COLUMN ... TYPE ...` would
silently break a chain and the next verification would report a row content
mismatch that was not tampering.

### The enforcement point

Four paths can change what a chain covers, and each needs the same lock
consulted: the DML hook (DELETE, UPDATE), `drop_table_in_schema` (DROP),
the truncate handler (TRUNCATE), and the shadow rewrite (a rewriting type
change). Three of the four have had the same hole at some point, each found
separately. They belong behind one resolver so a fifth path has an obvious
place to ask.

Behind those, the commit itself holds the line. A transaction that stamped
rows deleted in a table whose commits are chained is refused at commit
whatever statement got the delete past the lock, because a chain covers rows
that stay.

### Row order

A commit's rows are hashed in the order they are written, and a verification
reads them back in the order they are stored, ascending `(page_num, slot)`
through `TupleId::order_key`. The two are kept equal by the transaction's own
insertion cursor.

The heap places single-page inserts on the writing thread's tail page. A
transaction whose statements run on different worker threads, or one
statement whose batches are polled on different threads, would then store a
later run below an earlier one. So a chained table's rows go through
`HeapFile::insert_batch_with_cursor` with one cursor per table per
transaction, handed out by `PendingChainWrites::cursor`. The cursor starts
from the writing thread's tail page, so a small transaction shares a page the
way a thread's inserts do, and only ever moves forward: a page that fills
rolls to a fresh page rather than to one reclaimed space was found on. The
thread's own tail follows it.

Each batch still produces one run of contiguous ascending ids, and
`PendingChainWrites::record` takes the run's first and last position. A run
that starts at or below the last position already taken would mean the running
hash covers a sequence no read can reproduce, so it is recorded and the commit
is refused. The cursor is what keeps that from happening; refusing is what
keeps a silent mismatch from becoming a false verification failure if it ever
did.

## The lock scope

`CommitChain::append` holds `ChainState` for:

1. Read `linking.head_hash` and `linking.commits`, and the version.
2. `CommitHash::link`, one SHA-256 over the entry's fields.
3. Push onto `pending`, advance `linking`.

That is the whole serial section. Not in it: hashing the rows (before), the
WAL record (after, on the caller's side), the commit record, the durability
wait, and the file write.

`ChainState` carries two heads:

- `linking`, what the next entry links onto, pending entries included.
- `durable`, what the chain holds. `head()` returns this, so a verification, an
  anchor and a view all see only entries whose transaction committed.

## Pending entries, and why the file is written at commit

The chain file only ever takes entries whose transaction committed.

An entry is linked before the commit record is written, so at that moment
nobody knows whether the transaction will commit. Writing it to the file
immediately produced the defect the kill-point test found: a process stopped
between the link and the commit left an entry in the file for a transaction
that never happened, and the next start read it as the head.

So `append` pushes onto `pending`, and `publish(txn_id)` writes to the file
once the commit record is written. `publish` drains the pending list from the
front while the front is committed, so the file is always a whole chain from
its first entry to its last, never one with a hole where an earlier
transaction has not finished. `discard(txn_id)` drops a transaction's pending
entries and rolls `linking` back; discarding one with entries linked after it
is refused, because those chained onto it.

The file is written through a buffer of `WRITE_BUFFER_RECORDS` records over
a plain write handle held at the end of the file. A published record is a
copy into that buffer, and the buffer reaches the file when it fills, before
anything reads the chain, and when the chain is flushed. Reads go through a
handle of their own, opened on the first read and kept, so a walk, an anchor
check and a version range open nothing and block no commit. Durability is
the write-ahead log, not the file. Ahead of the commit record, in the
transaction's own log chain:

```text
LogRecordType::CommitChainEntry = 21
payload: table_id (u32) | sequence (u64) | the 96-byte record
```

Recovery collects these into `RecoveryResult::chain_records` and filters them
to committed transactions the same way `redo_records` are filtered. At start,
`restore_chain_entries` lays them back into their chains through `adopt`,
which recomputes each entry's link against the head the chain stands at so an
entry that does not belong there fails the start rather than being written,
and passes over an entry already held, so replaying a log twice changes
nothing.

Because a chain record is dropped with its transaction's commit record at a
checkpoint boundary, `ChainRegistry::sync_all` puts the open chains on the
device before the boundary is recorded.

## On a consensus group

The hash travels in the changeset and every member links the entry as it
applies the changeset:

```text
OP_COMMIT_CHAIN = 18
payload: table_id (u32) | rows_hash (32 bytes) | row_count (u64) | algorithm_id (u16)
```

The connection that ran the transaction hashes its rows and, on a group,
links nothing. Its commit captures one op per table and proposes. When the
entry applies, in the log's order on every member:

- The leader's own transaction is finished by `ChangesetMachine::finish_local`,
  which links the carried hash onto this node's chain under that transaction,
  logs the record into the transaction's log chain, writes the commit record,
  and publishes.
- Every other member stages the transaction, applies its rows through the
  same `InsertOperator` with the staged transaction's own accumulator on the
  context, and at the final chunk holds what it hashed against what the op
  carries. A member whose rows hashed differently, or that applied none, is
  holding different data and says so rather than chaining. The entry is then
  linked under the staged transaction, logged, committed and published.

`commit_version` is the entry's index and `commit_ts` its proposal instant,
the same on every member, and the head is whatever each member's chain stands
at, which the log's order makes the same as well. So every member computes
the same link at the same position for the same commit, while stamping the
rows with a transaction id of its own, which the link does not cover.

Carrying the hash rather than a link the leader computed is what keeps two
commits proposed together from linking in one order and applying in the
other: the group decides the order the commits apply in, and that is the
order they link in on every member, whichever connection reached its commit
first.

`COMMIT_CHAINS_INTRODUCED_IN` gates it. While a member of the group runs a
binary that does not read the tag, a write to a verified table is refused
rather than chained on the other members alone, because a chain that advanced
there would leave that member unable to link any later entry.

`VERIFY TABLE` is classified `Local`. Each member answers for the copy it
holds, from its own chain and its own rows, and records the run in its own
history.

## The genesis entry

Enabling verification on a populated table writes one entry over the rows
already there, flagged as the genesis in the record. Its `txn_id` field names
the transaction id fence the table was chained at rather than a transaction,
and its `rows_hash` is over the set of rows rather than a sequence of them:
each row's digest under the canonical encoding, sorted ascending, hashed in
that order (`SetHasher`). A set larger than one in-memory run is sorted in
runs under `<data_dir>/verify/` and merged, so the memory the set takes is
bounded whatever the table holds.

A set rather than a sequence because the members of a group store the same
rows in different orders, and the set is what they share. Every member's
genesis is therefore the same entry, at the same position, over the same
rows.

The order the statement takes on every member:

1. The registry fences the table at the next transaction id
   (`ChainRegistry::chain_from`). From here a transaction at or above the
   fence hashes what it writes to the table.
2. The catalog records the table as verified.
3. The rows the snapshot sees stamped below the fence are hashed into the
   set, and the entry is appended, published and flushed. On a member of a
   group the entry takes the index and instant of the log entry the
   statement runs under.
4. The catalog records where the entry landed (`genesis_at`, one past its
   position), because on a node that leads no group a commit can link ahead
   of it.

A transaction from below the fence that had written to the table, hashed or
not, or that had stamped rows deleted in it, is refused at its commit: its
rows are covered by neither the set nor an entry of their own. That is what
makes what the snapshot sees below the fence exactly the set, without waiting
for any transaction to end, so the statement holds up no commit on a member
of a group.

A table with no rows gets no genesis entry: there is nothing to cover as a set,
and its chain begins at its first commit.

A sampled verification checks the genesis entry's link and does not read the
set back, because reading it back is reading the whole of what the table
held, and says so in its summary. A full verification reads it back under its
own snapshot, every row the snapshot sees stamped below the fence.

`zyron_sys.compliance.log` is registered as a verified table at every start,
so the first start that finds its chain empty writes the genesis over the rows
the log holds (`cover_compliance_log`), before anything appends to it.

## Anchors

`AnchorStore` holds the latest anchor per table at
`<data_dir>/verify/anchors.zvan` (`FormatKind::VerificationAnchor`, magic
`ZVAN`), rewritten whole under its lock. One anchor per table rather than a
history: an anchor a shorter chain contradicts is contradicted whichever is
compared, and the latest is the one a truncation has to get past.

`anchor_table` syncs the chain before recording the head, so an anchor never
names a head the chain cannot show. It writes a `ChainAnchored` event into the
audit log through the same `append_audit` every other event goes through,
which means anchoring the compliance log's own chain extends that chain.

`background/chain_anchor.rs` runs the pass on `verify.anchor_interval_secs`,
anchoring every verified table whose head has moved since its last anchor in
one write of the store, and raising `chain_not_anchored` for a head that has
gone twice the interval without one.

An exported anchor adds a signature over `Anchor::signed_bytes` under the
release signing scheme. `check_exported_anchor` answers two things separately:
whether the signature holds and whether the artifact names the head the chain
shows. Both have to hold. An artifact that matches the chain and carries no
valid signature was written by whoever rewrote the chain.

## The walk

`verify::walk` takes the chain, a `CommitRows`, a range, a mode, a sample, the
anchors and a cancellation check.

The range is read in windows of `REHASH_WINDOW_COMMITS` entries, each window
in bulk reads of `READ_WINDOW_RECORDS` records, so a chain of millions of
entries is walked with a bounded number of them in memory and a few large
reads rather than one per record.

Link checking runs for every entry in range whatever the mode: `prev_hash`
against the previous entry's `entry_hash`, and `entry_hash` against
`compute_entry_hash`. That is the cheap half and it catches an edited entry on
its own.

Row rehashing runs for every commit under `RowMode::All` and for
`sample_positions` under `RowMode::Sampled`, which spreads the sample across
the range with both ends included.

`CommitRows::hash_commits` takes every transaction of a window to be rehashed,
and the genesis fence when the set is to be read back, and answers in **one
pass** over the table. `HeapCommitRows` walks the heap in windows of
`SCAN_WINDOW_PAGES` pages through `HeapFile::scan_window`, pinning the pages
the pool holds and carrying the rest without putting them in the pool, so a
table larger than the pool is read with a bounded footprint and a
verification evicts nothing the statements the node is serving are using.
Cancellation is checked between pages. One `RowsHasher` per named
transaction takes each row its `xmin` names. A commit's rows sit together,
so the transaction a row belongs to is nearly always the one the row before
it belonged to, and whether that transaction is wanted or not, the answer
for the row before is the answer for this one without touching the map.
The genesis set takes every row the pass's snapshot sees stamped below the
fence. A sampled verification of a table with a hundred thousand commits
would otherwise read the table a hundred thousand times.

Anchors are checked last, because an anchor a shorter chain contradicts is the
case a walk of the chain alone cannot see.

Failures are typed (`VerifyFailure`) and name the first commit that did not
match: `RowContentMismatch`, `RowCountMismatch`, `BrokenLink`,
`EntryHashMismatch`, `MissingCommit`, `AnchorContradiction`.

The walk runs on a blocking thread charged to `WorkloadClass::Background`, so
a verification never takes throughput from the statements the node is serving.
It takes no locks on the table. The read runs under a transaction of its own,
so the pass sees one snapshot and the rows it reads are not reclaimed under
it. `VERIFY TABLE` publishes a context of its own under the connection's
backend id and secret, so a cancel request or `CANCEL BACKEND` stops the walk
at the next page.

## The compliance log's conversion

`zyron_sys.compliance.log` carried a CRC32 chain in two fields of every row:
`prev_hash` and `entry_hash`, computed by `ComplianceLogEntry::compute_hash`
and walked by `Catalog::verify_compliance_chain`. All three are deleted.

CRC32 is a checksum, not a one-way function. Anyone who could rewrite an entry
could recompute every hash after it in microseconds, so the chain stated only
that nobody had corrupted a row by accident. It was one chain implementation
too many and the weaker of the two.

The log is now a registered table of the system catalog
(`Catalog::register_compliance_log`), over the heap its rows were already in,
with `immutable` and `verified` set. `append_audit` writes the row and the
chain entry in one transaction, through the transaction's own cursor, so an
event that is in the log is covered by the chain and one that is not is in
neither. The event id comes from a counter read off the log's last row once
per process, so an append costs one row rather than a read of the whole log.

The row shape moved to `COMPLIANCE_LOG_SCHEMA_VERSION` V2 with a registered
`CatalogSchemaEvolution` step, `drop_compliance_log_hashes`. It is one way: the
entries written after the conversion never had the fields. Its description
records that the chain it removed was a checksum rather than evidence, so an
auditor reading history from before the conversion is told what it is worth
rather than being left to assume.

`system_compliance_report`'s `audit` kind runs `verify_dispatch::run_verify`
against the log, the same function `VERIFY TABLE` runs, so the report and the
statement answer with one result through one code path.

## Exactly one chain implementation

The phase's standing invariant, asserted by
`crates/zyron-catalog/tests/compliance_log_conversion_test.rs`:

- `verify_compliance_chain`, `ComplianceLogEntry::compute_hash`, `AuditChain`
  and the `audit_chain` module appear nowhere in `crates/`.
- No file outside the chain's own module builds a row with a `prev_hash` or
  `entry_hash` field.
- Nothing calls `hash32` on a line that also mentions a previous hash or a
  chain. The checksum is still the right tool for a page or a payload, and
  never for linking one record to the one before it.
- Exactly one file defines `compute_entry_hash`.

A hash-chained log added by a later phase reuses this. The mechanism is not
particular to compliance beyond which table the rows land in: set `immutable`
and `verified`, write through a commit, and `VERIFY TABLE` walks it.
