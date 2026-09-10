# xnpv

Discounts each flow by the time from the first date, measured in days over a 365-day year, so irregular spacing is handled where npv assumes even periods. The two arrays must hold the same number of entries.

## Syntax

```sql
xnpv(rate, dates, cashflows)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL or an array does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `dates` | JSON array of dates as days since 1970-01-01, one per cashflow. | Not applicable. |

## Examples

```sql
SELECT xnpv(0.1, '[0,365]', '[-100,110]')
```

Effectively 0, because 110 a year later is worth 100 at a 10 percent rate.

## Refused

- The two arrays differ in length.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [xirr](xirr.md)
- [npv](npv.md)
