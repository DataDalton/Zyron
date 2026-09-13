# ALTER CHANGE STREAM

Moves the stream's position, which is how a stale stream is recovered and how a consumer replays or skips changes. A reset to a version or an instant names a place in the feed, a reset to a position names it as the count of changes consumed, which is the form that reads the same on every member of a group, and a reset with no target moves to the oldest change the feed still holds. SET COLUMNS changes what the stream yields, which is also how a stream that needs attention after a column was dropped is corrected.

## Syntax

```sql
ALTER CHANGE STREAM name RESET [TO VERSION n | TO TIMESTAMP 'ts' | TO POSITION n | TO EARLIEST | TO LATEST] | SET COLUMNS (col, ...) | SET ALL COLUMNS
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `RESET` | Moves the position to the oldest change the feed still holds. | Not applicable. |
| `RESET TO VERSION n | RESET TO TIMESTAMP 'ts'` | Moves the position to that version or instant, so the next read yields the changes past it. | Not applicable. |
| `RESET TO POSITION n` | Moves the position to the count of changes consumed, which names the same change on every member of a group. | Not applicable. |
| `RESET TO LATEST` | Moves the position to the source's current version, so the next read yields nothing until the next change. | Not applicable. |
| `SET COLUMNS (col, ...) | SET ALL COLUMNS` | Changes the columns the stream yields, and clears the attention a dropped column raised. | Not applicable. |

## Examples

```sql
ALTER CHANGE STREAM order_changes RESET TO VERSION 1200
```

The next read yields the changes after version 1200.

## Refused

- The version or instant named is below what the feed still holds.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE CHANGE STREAM](create-change-stream.md)
- [SHOW CHANGE STREAMS](show-change-streams.md)
