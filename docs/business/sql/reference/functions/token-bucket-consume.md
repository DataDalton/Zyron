# token_bucket_consume

Refills the bucket for the time since it was last touched, then takes the tokens if enough are present. The result holds both an allowed flag and the updated bucket, so the new bucket has to be stored for the limit to hold across calls. Nothing is persisted by this function.

## Syntax

```sql
token_bucket_consume(bucket, tokens, now)
```

## Returns

BYTEA holding the allowed flag and the updated bucket as JSON. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `tokens` | How many tokens the request costs. | Not applicable. |
| `now` | Current time as epoch microseconds, which drives the refill. | Not applicable. |

## Examples

```sql
SELECT token_bucket_consume(token_bucket_create(10, 1), 3, 0)
```

An object with allowed true and a bucket holding 7 tokens.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [token_bucket_create](token-bucket-create.md)
- [token_bucket_available](token-bucket-available.md)
- [leaky_bucket_add](leaky-bucket-add.md)
