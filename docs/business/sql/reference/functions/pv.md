# pv

Discounts the level payments and the final value back to the present. It inverts fv, so the two agree on the same rate, period count and payment.

## Syntax

```sql
pv(rate, nper, pmt, fv)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `fv` | Amount remaining at the end of the last period. | Not applicable. |

## Examples

```sql
SELECT pv(0.01, 12, -100, 0)
```

Approximately 1125.51, what twelve payments of 100 are worth today.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [fv](fv.md)
- [pmt](pmt.md)
- [npv](npv.md)
