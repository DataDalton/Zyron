# variance_pop

Averages the squared deviations from the mean over the value count. The units are those of the input squared, so stddev_pop is the comparable figure. An empty array gives 0.0.

## Syntax

```sql
variance_pop(values)
```

## Returns

DOUBLE PRECISION. NULL when the array is NULL or holds a non-number.

## Examples

```sql
SELECT variance_pop('[2,4,4,4,5,5,7,9]')
```

4.

## Refused

- The argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [variance_sample](variance-sample.md)
- [stddev_pop](stddev-pop.md)
