# leaky_bucket_add

Drains the bucket for the time since it was last touched, then adds the amount if the capacity allows. The result holds both an accepted flag and the updated bucket, so the new bucket has to be stored for the limit to hold across calls.

## Syntax

```sql
leaky_bucket_add(bucket, amount, now)
```

## Returns

BYTEA holding the accepted flag and the updated bucket as JSON. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `amount` | How much the request adds to the level. | Not applicable. |

## Examples

```sql
SELECT leaky_bucket_add(leaky_bucket_create(10, 1), 11, 0)
```

An object with accepted false, because 11 is beyond the capacity.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [leaky_bucket_create](leaky-bucket-create.md)
- [token_bucket_consume](token-bucket-consume.md)
