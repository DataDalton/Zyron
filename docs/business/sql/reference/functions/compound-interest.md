# compound_interest

Raises one plus the rate over the compounding count to the power of the count times the years, and multiplies by the principal. The result is the total value rather than the interest alone, so subtract the principal for the interest earned.

## Syntax

```sql
compound_interest(principal, rate, n, t)
```

## Returns

DOUBLE PRECISION, the total value. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `n` | Compounding periods per year. | Not applicable. |
| `t` | Number of years. | Not applicable. |

## Examples

```sql
SELECT compound_interest(1000, 0.05, 12, 1)
```

Approximately 1051.16, a year of monthly compounding at five percent.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [fv](fv.md)
- [pv](pv.md)
