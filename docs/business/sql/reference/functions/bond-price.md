# bond_price

Discounts every coupon and the face value at the yield. A yield above the coupon rate puts the price below face, and a yield below it puts the price above face. Both rates are per period, so a semi-annual bond takes half the annual figures and twice the years.

## Syntax

```sql
bond_price(face, coupon_rate, yield_rate, periods)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `periods` | Coupon periods remaining. | Not applicable. |

## Examples

```sql
SELECT bond_price(1000, 0.05, 0.05, 10)
```

Effectively 1000, because the yield matches the coupon rate.

## Refused

- A parameter is zero or negative.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bond_yield](bond-yield.md)
- [npv](npv.md)
