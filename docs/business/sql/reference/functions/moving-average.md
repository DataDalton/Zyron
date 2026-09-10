# moving_average

Averages each position with the values before it, up to the window size. The first positions average the fewer values available rather than returning NULL, so the output is the same length as the input and its start is noisier than its body. A window of 0 returns the input unchanged.

## Syntax

```sql
moving_average(values, window)
```

## Returns

ARRAY as JSON text, the same length as the input. NULL when the array or the window is NULL, the array holds a non-number, or the window is negative.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `window` | How many values each average covers, counting the current one. | Not applicable. |

## Examples

```sql
SELECT moving_average('[1,2,3,4]', 2)
```

[1.0,1.5,2.5,3.5], where the first position averages one value.

## Refused

- Called with any count of arguments other than two.
- The values argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [weighted_moving_average](weighted-moving-average.md)
- [exponential_smoothing](exponential-smoothing.md)
