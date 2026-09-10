# outlier_detect_iqr

Takes the first and third quartiles by the same interpolation percentile uses, then flags every value strictly below the first quartile minus the factor times the range, or strictly above the third quartile plus it. Quartiles move less than a mean under extreme values, so this flags a lone extreme value the z score form can miss.

## Syntax

```sql
outlier_detect_iqr(values, factor)
```

## Returns

ARRAY of booleans as JSON text, the same length as the input. NULL when the array or the factor is NULL, or the array holds a non-number.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `factor` | Multiple of the interquartile range the fences sit beyond the quartiles. 1.5 is the common choice, and 3.0 flags only far values. | Not applicable. |

## Examples

```sql
SELECT outlier_detect_iqr('[1,2,3,4,100]', 1.5)
```

[false,false,false,false,true].

## Refused

- The values argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [outlier_detect_zscore](outlier-detect-zscore.md)
- [percentile](percentile.md)
