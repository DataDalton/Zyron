# weighted_moving_average

Applies the first weight to the current value, the second to the value before it, and so on back through the weight array. Each output divides by the sum of the weights actually applied, so the first positions are rescaled rather than pulled toward zero by missing history. The weights need not sum to 1.

## Syntax

```sql
weighted_moving_average(values, weights)
```

## Returns

ARRAY as JSON text, the same length as the input. NULL when either array is NULL, holds a non-number, is empty, or the weights sum to zero.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `weights` | JSON array of weights, newest position first. Its length sets the window. | Not applicable. |

## Examples

```sql
SELECT weighted_moving_average('[1,2,3]', '[0.5,0.5]')
```

[1.0,1.5,2.5].

## Refused

- Either argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [moving_average](moving-average.md)
- [exponential_smoothing](exponential-smoothing.md)
