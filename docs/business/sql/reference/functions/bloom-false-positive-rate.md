# bloom_false_positive_rate

Counts the set bits and raises the set fraction to the power of the hash count. The result reflects how full the filter is now, not the rate bloom_create was asked for, so it rises with every addition and reaches the requested rate at about the expected item count. An empty filter gives 0.

## Syntax

```sql
bloom_false_positive_rate(filter)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when the filter is NULL or its bytes do not parse.

## Examples

```sql
SELECT bloom_false_positive_rate(bloom_create(1000, 0.01))
```

0, because no bits are set.

## Refused

- The filter argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bloom_create](bloom-create.md)
- [bloom_add](bloom-add.md)
- [hll_error](hll-error.md)
