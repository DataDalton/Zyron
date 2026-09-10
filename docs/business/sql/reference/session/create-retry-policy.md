# CREATE RETRY POLICY

Sets how a failed operation is retried: the attempt limit and the wait between attempts. An exponential backoff lengthens the wait after each attempt, bounding the request rate against a failing dependency.

## Syntax

```sql
CREATE RETRY POLICY name (max_attempts = n, backoff = 'exponential' | ..., ...)
```

## Examples

```sql
CREATE RETRY POLICY ingest (max_attempts = 3, backoff = 'exponential')
```

A named policy that operations are retried under.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP RETRY POLICY](drop-retry-policy.md)
- [CREATE BULKHEAD](create-bulkhead.md)
