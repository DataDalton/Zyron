# CREATE CDC INGEST

Reads a change feed from another system and applies it into a table, which is the other direction from CREATE CDC STREAM. The options name the feed and the key the changes are applied by, because applying a change needs to know which row it is about.

## Syntax

```sql
CREATE CDC INGEST name FROM source INTO table WITH (option = value, ...)
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH (primary_key = 'col', ...)` | Names the feed and the key each change identifies its row by. | Not applicable. |

## Examples

```sql
CREATE CDC INGEST i FROM kafka INTO orders WITH (topic = 'evt')
```

Changes from that feed are applied into the table.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP CDC INGEST](drop-cdc-ingest.md)
- [CREATE CDC STREAM](create-cdc-stream.md)
