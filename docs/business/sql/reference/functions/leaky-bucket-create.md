# leaky_bucket_create

Returns a bucket at level zero with the given capacity and leak rate. A leaky bucket fills as work arrives and drains at a constant rate, so it smooths a burst into a steady flow where a token bucket lets the burst straight through.

## Syntax

```sql
leaky_bucket_create(capacity, leak_rate)
```

## Returns

BYTEA holding the bucket. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `capacity` | Most the bucket can hold before work is refused. | Not applicable. |
| `leak_rate` | Amount drained per second. | Not applicable. |

## Examples

```sql
SELECT leaky_bucket_add(leaky_bucket_create(10, 1), 3, 0)
```

An object with accepted true and a bucket at level 3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [leaky_bucket_add](leaky-bucket-add.md)
- [token_bucket_create](token-bucket-create.md)
