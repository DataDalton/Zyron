# CREATE EXTERNAL SOURCE

Names an external location rows can be read from, so a query or job refers to it by name instead of repeating a URI and format. Credentials are held against the source and do not appear in statement text or logs.

## Syntax

```sql
CREATE EXTERNAL SOURCE [IF NOT EXISTS] name TYPE backend URI 'uri' FORMAT fmt [MODE ...] [OPTIONS (...)] [CREDENTIALS (...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `CREDENTIALS (k = v, ...)` | The secret the source authenticates with, held against the source. | The backend's ambient credentials are used. |
| `OPTIONS (k = v, ...)` | Backend options, such as a delimiter or a compression. | The format's own defaults apply. |

## Examples

```sql
CREATE EXTERNAL SOURCE src TYPE FILE URI '/tmp/src' FORMAT CSV
```

A named place rows can be read from.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER EXTERNAL SOURCE](alter-external-source.md)
- [DROP EXTERNAL SOURCE](drop-external-source.md)
- [CREATE EXTERNAL SINK](create-external-sink.md)
