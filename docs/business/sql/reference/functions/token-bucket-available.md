# token_bucket_available

Refills for the time since the bucket was last touched and reports the count, without changing the bucket. Use it to show headroom, and token_bucket_consume to take it.

## Syntax

```sql
token_bucket_available(bucket, now)
```

## Returns

DOUBLE PRECISION. NULL when either argument is NULL or the bucket does not parse.

## Examples

```sql
SELECT token_bucket_available(token_bucket_create(10, 1), 0)
```

10.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [token_bucket_consume](token-bucket-consume.md)
- [token_bucket_create](token-bucket-create.md)
