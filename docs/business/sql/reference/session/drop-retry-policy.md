# DROP RETRY POLICY

Removes the policy. Operations that named it fall back to whatever the surrounding default is, which may be no retry at all.

## Syntax

```sql
DROP RETRY POLICY name
```

## Examples

```sql
DROP RETRY POLICY ingest
```

Operations that named it fall back to the default.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE RETRY POLICY](create-retry-policy.md)
