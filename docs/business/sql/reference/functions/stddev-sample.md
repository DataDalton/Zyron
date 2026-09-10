# stddev_sample

Square root of the sample variance, over a denominator of one less than the value count. Correct when the array is a sample drawn from a larger population, where the population form understates the spread. Fewer than two values gives 0.0.

## Syntax

```sql
stddev_sample(values)
```

## Returns

DOUBLE PRECISION. NULL when the array is NULL or holds a non-number.

## Examples

```sql
SELECT stddev_sample('[2,4,4,4,5,5,7,9]')
```

Approximately 2.138, above the population figure of 2.

## Refused

- The argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [stddev_pop](stddev-pop.md)
- [variance_sample](variance-sample.md)
- [correlation](correlation.md)
