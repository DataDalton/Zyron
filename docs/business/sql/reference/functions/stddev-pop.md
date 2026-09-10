# stddev_pop

Square root of the population variance, over a denominator of the value count. Correct when the array holds every member of the population rather than a sample drawn from one. An empty array gives 0.0.

## Syntax

```sql
stddev_pop(values)
```

## Returns

DOUBLE PRECISION. NULL when the array is NULL or holds a non-number.

## Examples

```sql
SELECT stddev_pop('[2,4,4,4,5,5,7,9]')
```

2.

## Refused

- The argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [stddev_sample](stddev-sample.md)
- [variance_pop](variance-pop.md)
- [zscore](zscore.md)
