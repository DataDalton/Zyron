# ALTER ENDPOINT

Changes an endpoint's configuration, or enables and disables it. A disabled endpoint answers that it is unavailable rather than disappearing, which tells a caller the path exists and is off rather than that it was never there.

## Syntax

```sql
ALTER ENDPOINT name SET ... | ENABLE | DISABLE
```

## Examples

```sql
ALTER ENDPOINT recent DISABLE
```

The path answers that it is unavailable.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ENDPOINT](create-endpoint.md)
