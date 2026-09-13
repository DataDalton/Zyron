# DROP CHANGE STREAM

Removes the stream. The changes it was positioned over stay in the feed for their retention, and the grants on the stream are forgotten.

## Syntax

```sql
DROP CHANGE STREAM [IF EXISTS] name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IF EXISTS` | Succeeds when there is no such stream. | A missing stream is an error. |

## Examples

```sql
DROP CHANGE STREAM order_changes
```

The stream and its position are gone.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE CHANGE STREAM](create-change-stream.md)
