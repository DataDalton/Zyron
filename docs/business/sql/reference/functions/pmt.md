# pmt

Returns the level payment that clears the present value over the given number of periods at the given rate. The rate is per period, so an annual rate on monthly payments has to be divided by 12 and the period count multiplied by it. A rate of 0 gives the principal spread evenly.

## Syntax

```sql
pmt(rate, nper, pv)
```

## Returns

DOUBLE PRECISION, negative for money paid out. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `nper` | Number of payment periods. | Not applicable. |
| `pv` | Amount borrowed now. | Not applicable. |

## Examples

```sql
SELECT pmt(0.01, 12, 1000)
```

Approximately -88.85, the monthly payment on 1000 at one percent a month.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [pv](pv.md)
- [fv](fv.md)
- [amortization_schedule](amortization-schedule.md)
