# forecast_linear

Fits the same line as linear_regression and evaluates it at each value in future_x, returning one number per projection. The fit carries no confidence bound and no curvature, so a projection far past the fitted range holds only as far as the straight line does.

## Syntax

```sql
forecast_linear(x, y, future_x)
```

## Returns

ARRAY as JSON text, one number per future x. NULL when any array is NULL, holds a non-number, the two fitted arrays differ in length, they hold fewer than two values, or every x is the same.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `future_x` | The x values to project at. Length is independent of the fitted series. | Not applicable. |

## Examples

```sql
SELECT forecast_linear('[1,2,3]', '[2,4,6]', '[4,5]')
```

[8.0,10.0].

## Refused

- Called with any count of arguments other than three.
- Any argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [linear_regression](linear-regression.md)
- [exponential_smoothing](exponential-smoothing.md)
- [moving_average](moving-average.md)
