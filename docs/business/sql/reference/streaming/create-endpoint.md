# CREATE ENDPOINT

Publishes one statement at an HTTP path. The statement is fixed when the endpoint is created. A caller supplies only its parameters and cannot substitute another statement.

## Syntax

```sql
CREATE ENDPOINT [IF NOT EXISTS] name ON PATH '/path' METHOD GET [, ...] USING 'sql' AUTH NONE | JWT | ... [REQUIRE SCOPE 'name' [, ...]] [RATE LIMIT n]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `AUTH NONE | JWT` | How a caller is authenticated before the statement runs. | Not applicable. |
| `REQUIRE SCOPE 'name' [, ...]` | Scopes a caller's token must carry. | No scope beyond authentication is required. |
| `RATE LIMIT n` | Bounds how often a caller may run it. | The surface's own limits apply. |

## Examples

```sql
CREATE ENDPOINT recent ON PATH '/recent' METHOD GET USING 'SELECT id FROM orders' AUTH NONE
```

A path that runs that statement when it is called.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER ENDPOINT](alter-endpoint.md)
- [DROP ENDPOINT](drop-endpoint.md)
- [CREATE STREAMING ENDPOINT](create-streaming-endpoint.md)
