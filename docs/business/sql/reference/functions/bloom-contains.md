# bloom_contains

Returns true when every bit the value addresses is set. A true result can be a false positive at the filter's current rate. A false result is exact, because a value that was added always leaves its bits set. A NULL filter, a NULL value, and filter bytes that do not parse all read as false rather than NULL.

## Syntax

```sql
bloom_contains(filter, value)
```

## Returns

BOOLEAN, never NULL.

## Examples

```sql
SELECT bloom_contains(bloom_add(bloom_create(100, 0.01), 'alice'), 'bob')
```

false, unless the two values collide at the filter's rate.

## Refused

- Either argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bloom_add](bloom-add.md)
- [bloom_false_positive_rate](bloom-false-positive-rate.md)
- [cms_estimate](cms-estimate.md)
