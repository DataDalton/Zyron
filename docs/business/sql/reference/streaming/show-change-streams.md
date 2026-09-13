# SHOW CHANGE STREAMS

Lists every change stream, or the streams over one table, or one stream by name, with the source tables, the position, the changes pending past it, the age of the oldest pending change, and whether the stream is stale or needs attention. The same columns are in zyron_sys.cdc.change_streams.

## Syntax

```sql
SHOW CHANGE STREAMS [ON TABLE table] | SHOW CHANGE STREAM name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ON TABLE table` | Lists the streams over that table alone. | Every stream is listed. |
| `SHOW CHANGE STREAM name` | Lists that stream alone. | Not applicable. |

## Examples

```sql
SHOW CHANGE STREAMS ON TABLE orders
```

One row per stream over the table, with its position and what pends past it.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [CREATE CHANGE STREAM](create-change-stream.md)
- [ALTER CHANGE STREAM](alter-change-stream.md)
