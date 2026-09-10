# outlier_detect_zscore

Returns one boolean per value, true where the absolute z score against the population standard deviation is strictly above the threshold. Mean and deviation come from the same array, so a single extreme value inflates both and can mask itself. A series with no variance gives all false.

## Syntax

```sql
outlier_detect_zscore(values, threshold)
```

## Returns

ARRAY of booleans as JSON text, the same length as the input. NULL when the array or the threshold is NULL, or the array holds a non-number.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `threshold` | Standard deviations from the mean past which a value is flagged. 3.0 is the common choice. | Not applicable. |

## Examples

```sql
SELECT outlier_detect_zscore('[1,2,3,100]', 1.5)
```

[false,false,false,true].

## Refused

- The values argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [outlier_detect_iqr](outlier-detect-iqr.md)
- [zscore](zscore.md)
- [stddev_pop](stddev-pop.md)
