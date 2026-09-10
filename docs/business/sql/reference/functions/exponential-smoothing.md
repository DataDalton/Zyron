# exponential_smoothing

Emits the first value unchanged, then each later position as alpha times the value plus one minus alpha times the previous smoothed result. Weight on older values therefore decays geometrically rather than dropping off at a window edge. Alpha is clamped to between 0.0 and 1.0, where 1.0 returns the input unchanged and 0.0 holds the first value throughout. Output length matches the input.

## Syntax

```sql
exponential_smoothing(values, alpha)
```

## Returns

ARRAY as JSON text, the same length as the input. NULL when the array or alpha is NULL, or the array holds a non-number.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `alpha` | Weight on the current value, between 0.0 and 1.0. Lower values smooth harder and lag further. | Not applicable. |

## Examples

```sql
SELECT exponential_smoothing('[10,20,30]', 0.5)
```

[10.0,15.0,22.5].

## Refused

- The values argument is not text or binary holding a JSON array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [moving_average](moving-average.md)
- [weighted_moving_average](weighted-moving-average.md)
- [forecast_linear](forecast-linear.md)
