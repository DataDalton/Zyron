# Change Streams

A table with its change data feed on records every change made to it. A change stream is a durable position over that record. Reading the stream inside a transaction yields the changes past the position, and the position moves when the transaction commits. A transaction that rolls back leaves the position where it was, so the same changes are handed out again on the next read.

## What this means for you

- Turn the feed on with `ALTER TABLE t SET (change_data_feed = true)`. From then on every insert, update, delete and truncate of the table is recorded, with the row's columns and the metadata of the change.
- `table_changes(t, from, to)` reads a range of those changes as rows, with no position and no lock. It is the batch shape.
- `CREATE CHANGE STREAM s ON TABLE t` holds a position. `SELECT ... FROM s` inside a transaction reads what pends and moves the position at commit. It is the continuous shape.
- A read of a stream is part of whatever transaction reads it. A load that fails after reading leaves the position untouched, and the retry reads the same changes. Reprocessing needs no bookkeeping of your own.
- Two transactions reading one stream wait on each other, so a change is handed out once.
- A stream that falls behind what the feed still holds is stale, and every read of it says so rather than skipping what was reclaimed.

## What the feed records

Each change is one record carrying the row's columns and these metadata columns.

| Column | What it holds |
| --- | --- |
| `_change_type` | `insert`, `update_preimage`, `update_postimage` or `delete`. A truncate is recorded as its own kind with no row. |
| `_commit_version` | The version the change was committed at. Higher is later. |
| `_commit_ts` | The instant of the commit. |
| `_commit_txn_id` | The transaction that made the change. |
| `_change_ordinal` | The change's position within its commit, so a commit's changes read back in the order they were made. |
| `_source_table` | On a stream over several tables, the table the change came from. |

An update is two records, the row before and the row after, adjacent under one commit. `cdf_before_image = false` records the row after alone, which halves an update-heavy feed and leaves nothing to read as the preimage.

The feed options are set on the table.

```sql
ALTER TABLE orders SET (
    change_data_feed = true,
    cdf_retention = '7 days',
    cdf_columns = 'id, status, total',
    cdf_before_image = true,
    cdf_compression = 'lz4'
)
```

| Option | What it does |
| --- | --- |
| `change_data_feed` | Records changes when on. Turning it off marks every stream over the table stale. |
| `cdf_retention` | How long a change is held. The default is seven days. A change older than this is reclaimed, and a stream positioned below it is stale. |
| `cdf_columns` | The columns the feed records, plus the table's key columns, so a change touching none of them still records the row's identity and shows that the row changed. Each change holds those columns and nothing else, so a wide table's feed is as narrow as the list. A read of any other column is refused naming it. Changing the list applies from the next change on, and a change recorded under an earlier list is read through that list, so a column added to the list is refused by name over a range holding changes recorded before it. |
| `cdf_before_image` | Whether an update records the row it replaced. |
| `cdf_compression` | The codec sealed feed segments are written with, `none`, `lz4` or `zstd`. |

Every feed on the node is listed in `zyron_sys.cdc.feeds` with its settings, the versions it holds, the rows it records and the bytes it occupies, which is where a feed's storage is accounted for.

## Reading a range

```sql
SELECT id, total, _change_type, _commit_version
FROM table_changes(orders, 1200, LATEST)
ORDER BY _commit_version, _change_ordinal
```

The start is exclusive and the end inclusive, so a read that ends at version 1500 continues with `table_changes(orders, 1500, LATEST)`. Either bound may be a version, a timestamp string, `EARLIEST` or `LATEST`, and the named forms `start_version`, `end_version`, `start_timestamp` and `end_timestamp` read the same way. A range whose start retention has already reclaimed is refused naming the oldest version still held, rather than skipping what is gone.

A range wholly inside retention reads the same every time it is read. That is what makes a load written against it replayable.

## Holding a position

```sql
CREATE CHANGE STREAM order_changes ON TABLE orders
```

The stream starts at the table's current version, so its first read yields the changes made after it was created. `AT VERSION n` and `AT TIMESTAMP 'ts'` start it elsewhere, and `SHOW INITIAL ROWS` makes the first read yield every existing row as an insert before continuing from the feed, which seeds a target and keeps it current with one definition.

```sql
BEGIN;
INSERT INTO silver_orders SELECT id, total FROM order_changes WHERE _change_type <> 'delete';
COMMIT;
```

The insert reads the changes pending on the stream and writes them. At `COMMIT` the position moves past what was read, in the same commit as the rows written. A `ROLLBACK`, a statement error or a dropped connection leaves the position where it was. A `MERGE`, a `CALL` body, a `DO` block, a scheduled statement and a pipeline stage each run as one transaction of their own, and a stream any of them reads moves in that commit.

An update is two changes, the row before and the row after, and `MERGE` refuses a source that matches one target row twice. A `MERGE` over a stream that carries updates reads the stream through a derived table that keeps the row after, `USING (SELECT ... FROM order_changes WHERE _change_type <> 'update_preimage') AS src`, or is written as `APPLY CHANGES`, which takes the stream as it is.

`WITH (peek => true)` after the stream reads what pends without taking the position lock and without moving anything, for a dashboard or a check. `WITH (max_rows => n)` ends the read after about n changes at a boundary no transaction writes across, so a backlog drains over several transactions rather than one.

A stream created with `APPEND_ONLY` yields inserts alone. `WHERE` narrows the changes, reading the row after an insert or an update and the row before a delete. `COLUMNS (...)` narrows what is yielded.

`ON TABLES (a, b)` makes one stream over several tables. Every read ends at one boundary across them, so a transaction that wrote to both is wholly inside a read or wholly after it, and `_source_table` names where each change came from. `ON VIEW v` follows a view over one table, projected and narrowed the way the view is.

A stream may be read by a principal holding SELECT on it, and what that principal reads is governed the way the table's own rows are. Row security, masking and purpose apply to the change rows per reader, so two principals reading one stream see different rows while the position moves once.

## When a stream is stale

A stream is stale when its position names changes the feed no longer holds. That happens when retention reclaimed them, when a byte cap purged them, when the feed was turned off, or when the source table was dropped. A read of a stale stream is refused with `ChangeStreamStale` naming the position, the oldest change still held, and the reset that recovers it.

```sql
ALTER CHANGE STREAM order_changes RESET TO VERSION 1500
```

`RESET` with no target moves to the oldest change held, `RESET TO LATEST` to the source's current version, and `RESET TO POSITION n` to a count of changes consumed, which names the same change on every member of a group. A reset to a version retention has reclaimed is refused.

On a cluster every member holds the same changes and the same positions. A table created `USING ZYRONLAKE` ships its data files to every member with the commit that adds them, so a stream over a lake table reads the same changes on any member, and a scheduled statement, a retention expiry or an outbound delivery runs once for the cluster, on the leader, with its rows and its stream positions landing on every member.

A stream that names a column the table dropped needs attention rather than being stale. It keeps its position and yields nothing until `ALTER CHANGE STREAM s SET COLUMNS (...)` or `SET ALL COLUMNS` corrects it. A narrowing type change on a table with streams is refused unless written with `ACKNOWLEDGE STREAM BREAK`, which marks every stream over the table as needing attention.

The sweeper finds a stale stream before a reader does. `zyron_sys.cdc.change_streams` shows every stream with its position, the changes and versions pending, the age of the oldest pending change, and whether it is stale or needs attention. The alerts `cdc_stream_lag`, `cdc_stream_stale`, `cdc_stream_needs_attention` and `cdf_retention_pressure` fire on those conditions through the node's contact channels.

## Pipelines that run on change data

A pipeline stage may consume a stream, and a pipeline may run when a stream holds enough pending changes.

```sql
CREATE PIPELINE cdc ON CHANGE DATA FROM order_changes MIN ROWS 100 MAX WAIT 5 MINUTES AS (
    STAGE land (CONSUME CHANGES FROM order_changes MAX ROWS 10000 INTO bronze_orders),
    STAGE apply (APPLY CHANGES INTO dim_orders FROM order_changes KEYS (id) SEQUENCE BY _commit_version)
)
```

Each stage runs as one transaction that moves the stream's position, so a stage that fails leaves the position and the next run reads the same changes. The trigger reads the pending count and never moves the position itself. A stream created with `SHOW INITIAL ROWS` seeds the target on the first run and continues incrementally on the next with no change to the definition.

## Outbound delivery

`CREATE CDC STREAM s FROM CHANGE STREAM order_changes TO kafka WITH (...)` delivers a stream's changes to a sink. Delivery moves the stream's position once the sink has taken the records, so where delivery has got to is the position in `zyron_sys.cdc.change_streams`. An outbound stream created with `ON TABLE t` gets a change stream of its own, named `__cdc_` followed by the outbound stream's name, created and dropped with it.

Replication slots are a separate, wire-level surface for logical replication. A slot pins its own position in the write-ahead log, and a change stream reads the change feed. Neither wraps the other.

## Branches

A stream created on a branch reads the branch's changes and keeps a position of its own. A write on the branch is recorded for the branch, and a stream on the table sees nothing of it. Inside a branch, `table_changes` reads the table's history up to the version the branch was taken at and the branch's own after it. Merging a branch lands its rows as the table's changes and consumes the branch's streams with the branch. Dropping a branch drops the streams created on it.

## Limits

| Setting | What it bounds |
| --- | --- |
| `cdc.change_streams_per_table` | Streams over one table. A stream past the cap is refused naming it. The default is 64. |
| `cdc.cdf_max_bytes_per_table` | Bytes one feed holds. Past it the oldest changes are purged first, `cdf_retention_pressure` fires, and the table keeps accepting writes. |
| `cdc.cdf_max_retention_secs` | The longest `cdf_retention` a table may ask for. |
