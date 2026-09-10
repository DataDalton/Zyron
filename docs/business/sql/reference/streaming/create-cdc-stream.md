# CREATE CDC STREAM

Sends a table's changes to a sink as they are committed, rather than on a schedule. What arrives is the change itself, so a consumer applies rows rather than comparing snapshots.

## Syntax

```sql
CREATE CDC STREAM name ON table TO sink [WITH (option = value, ...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH (option = value, ...)` | Options the sink reads, such as where to write and how to batch. | The sink's own defaults apply. |

## Examples

```sql
CREATE CDC STREAM s ON orders TO kafka
```

The table's committed changes arrive at that sink.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP CDC STREAM](drop-cdc-stream.md)
- [CREATE CDC INGEST](create-cdc-ingest.md)
