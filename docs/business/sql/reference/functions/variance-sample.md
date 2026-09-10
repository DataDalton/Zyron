# variance_sample

Averages the squared deviations from the mean over one less than the value count, which removes the downward bias the population form carries on a sample. Fewer than two values gives 0.0.

## Syntax

```sql
variance_sample(values)
```

## Returns

DOUBLE PRECISION. NULL when the array is NULL or holds a non-number.

## Examples

```sql
SELECT variance_sample('[2,4,4,4,5,5,7,9]')
```

Approximately 4.571.

## Refused

- The argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [variance_pop](variance-pop.md)
- [stddev_sample](stddev-sample.md)
- [covariance](covariance.md)
