# npv

Discounts each cashflow by the rate raised to its position and adds the results. The first cashflow sits at period 0 and is not discounted, so a flow at the end of the first period belongs in the second position. Spacing is assumed even, and xnpv takes dated flows instead.

## Syntax

```sql
npv(rate, cashflows)
```

## Returns

DOUBLE PRECISION. NULL when either argument is NULL or the array does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `rate` | Discount rate per period, as a fraction. | Not applicable. |
| `cashflows` | JSON array of amounts, one per period, negative for money out. | Not applicable. |

## Examples

```sql
SELECT npv(0.1, '[-100,60,60]')
```

Approximately 4.13, so the flows beat a 10 percent rate.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [irr](irr.md)
- [xnpv](xnpv.md)
- [pv](pv.md)
