# DROP ENDPOINT

Removes an endpoint, so its path no longer answers. The statement it published is not stored anywhere else, so it goes with the endpoint.

## Syntax

```sql
DROP ENDPOINT [IF EXISTS] name
```

## Examples

```sql
DROP ENDPOINT recent
```

The path no longer answers.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ENDPOINT](create-endpoint.md)
