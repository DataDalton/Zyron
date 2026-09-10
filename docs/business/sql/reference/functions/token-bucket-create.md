# token_bucket_create

Returns a bucket holding its capacity in tokens, the refill rate, and the time it was last refilled. A token bucket allows a burst up to the capacity and then settles to the refill rate, where a leaky bucket admits no burst at all.

## Syntax

```sql
token_bucket_create(capacity, refill_rate)
```

## Returns

BYTEA holding the bucket. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `capacity` | Most tokens the bucket can hold, which bounds a burst. | Not applicable. |
| `refill_rate` | Tokens added per second. | Not applicable. |

## Examples

```sql
SELECT token_bucket_available(token_bucket_create(10, 1), 0)
```

10, because a new bucket is full.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [token_bucket_consume](token-bucket-consume.md)
- [token_bucket_available](token-bucket-available.md)
- [leaky_bucket_create](leaky-bucket-create.md)
