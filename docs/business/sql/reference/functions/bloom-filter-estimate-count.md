# bloom_filter_estimate_count

Estimates the count from the proportion of set bits, by the standard inversion of the fill formula. A saturated filter would make the logarithm infinite, so the set bit count is clamped one below the bit count and the result stays finite. The figure is never negative and is rounded to a whole count.

## Syntax

```sql
bloom_filter_estimate_count(filter)
```

## Returns

BIGINT, never negative. NULL when the filter is NULL or its bytes do not parse.

## Examples

```sql
SELECT bloom_filter_estimate_count(bloom_create(1000, 0.01))
```

0, because no bits are set.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bloom_create](bloom-create.md)
- [bloom_add](bloom-add.md)
- [hll_count](hll-count.md)
