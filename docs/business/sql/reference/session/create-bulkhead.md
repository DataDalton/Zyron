# CREATE BULKHEAD

Bounds how many statements of one kind run concurrently and how long the rest may wait. Work beyond the concurrency bound queues. Work beyond the queue size is refused rather than queued, so load is shed from one workload instead of slowing every workload.

## Syntax

```sql
CREATE BULKHEAD name (max_concurrent = n, max_wait = 'duration', queue_size = n)
```

## Examples

```sql
CREATE BULKHEAD reports (max_concurrent = 10, max_wait = '5s', queue_size = 100)
```

At most ten run at once, the rest wait up to five seconds, and beyond a hundred queued they are refused.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP BULKHEAD](drop-bulkhead.md)
- [CREATE RETRY POLICY](create-retry-policy.md)
