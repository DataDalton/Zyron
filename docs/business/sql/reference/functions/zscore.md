# zscore

Subtracts the mean from the value and divides by the standard deviation. All three arguments are plain numbers rather than arrays, so the mean and deviation come from whatever aggregate or stored statistic produced them. A standard deviation of 0 gives 0.0 instead of failing.

## Syntax

```sql
zscore(value, mean, stddev)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL.

## Examples

```sql
SELECT zscore(70, 60, 5)
```

2.

## Refused

- Called with any count of arguments other than three.
- Any argument is text, binary or an interval.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [stddev_pop](stddev-pop.md)
- [outlier_detect_zscore](outlier-detect-zscore.md)
