# DROP BULKHEAD

Removes the bound. Work previously held to it competes for the node without a concurrency limit.

## Syntax

```sql
DROP BULKHEAD name
```

## Examples

```sql
DROP BULKHEAD reports
```

That work is no longer bounded.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE BULKHEAD](create-bulkhead.md)
