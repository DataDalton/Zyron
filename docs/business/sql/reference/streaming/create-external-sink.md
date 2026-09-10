# CREATE EXTERNAL SINK

Names a place outside the database that rows can be written to, so a stream or a job refers to it by name. As with a source, the credentials are held against the sink rather than repeated in statement text.

## Syntax

```sql
CREATE EXTERNAL SINK [IF NOT EXISTS] name TYPE backend URI 'uri' FORMAT fmt [OPTIONS (...)] [CREDENTIALS (...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `CREDENTIALS (k = v, ...)` | The secret the sink authenticates with, held against the sink. | The backend's ambient credentials are used. |

## Examples

```sql
CREATE EXTERNAL SINK out TYPE FILE URI '/tmp/out' FORMAT CSV
```

A named place rows can be written to.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER EXTERNAL SINK](alter-external-sink.md)
- [DROP EXTERNAL SINK](drop-external-sink.md)
- [CREATE EXTERNAL SOURCE](create-external-source.md)
