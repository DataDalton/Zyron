# table_changes

Yields one row per recorded change of a table whose change data feed is on, carrying the row's columns and the metadata columns _change_type, _commit_version, _commit_ts, _commit_txn_id and _change_ordinal. _change_type is insert, update_preimage, update_postimage or delete. A bound is a version, a timestamp string, EARLIEST or LATEST. The start is exclusive and the end inclusive, so reading from the version a previous read ended at continues it. A predicate on the metadata columns is pushed into the read, so a range narrowed by version or by change type reads only the segments holding it.

## Syntax

```sql
table_changes(table [, start [, end]] | table, start_version => n | start_timestamp => 'ts', end_version => n | end_timestamp => 'ts', schema => 'current' | 'as_of_change')
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `start_version => n | start_timestamp => 'ts'` | Where the read starts, exclusive. EARLIEST reads from the oldest change the feed still holds. | EARLIEST. |
| `end_version => n | end_timestamp => 'ts'` | Where the read ends, inclusive. | LATEST. |
| `schema => 'as_of_change'` | Renders each change through the columns the table had when the change was written, dropped columns included. A column added later is absent from an older change. | Each change is rendered through the table's current columns. A column added after a change reads NULL in it, and a dropped column is not yielded. |

## Examples

```sql
SELECT * FROM table_changes(orders, 0, LATEST) ORDER BY _commit_version, _change_ordinal
```

Every recorded change of the table, oldest first, each with the row's columns and the change's metadata.

```sql
SELECT id, total FROM table_changes(orders, start_timestamp => '2026-09-01 00:00:00') WHERE _change_type = 'delete'
```

The rows deleted since that instant.

## Refused

- The table has no change data feed.
- The start is a version older than the feed's retention.
- An argument name is not one the function reads.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [CREATE CHANGE STREAM](create-change-stream.md)
- [ALTER TABLE SET OPTIONS](../lake/alter-table-set-options.md)
